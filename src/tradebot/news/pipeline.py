"""One news run: collect, drop what is known, score within budget, work out the risk level.

Scores are kept as Parquet in the ``data`` bucket. These files are not part of the nightly cache
sync, which would overwrite them with an older copy. ``first_seen`` is when the bot really knew
something, and it is the only honest clock for testing: a model cannot be backtested on old news,
because it may remember what happened next. So items older than a day are stored but never scored.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tradebot.config import NewsConfig
from tradebot.news.filters import relevant, same_story, suspicious, words
from tradebot.news.risk import assess
from tradebot.news.scoring import PROMPT_VERSION, Assessment, Provider, score
from tradebot.news.sources import Fetch, Item, Source, collect, download
from tradebot.store import Store

log = logging.getLogger(__name__)

BUCKET = "data"
SCORED_FILE = "news_scored.parquet"
RISK_FILE = "news_risk.parquet"

PENDING, SCORED, DUPLICATE, IGNORED, OLD, EXPIRED, SUSPICIOUS = (
    "pending", "scored", "duplicate", "ignored", "old", "expired", "suspicious")

MAX_AGE = pd.Timedelta("24h")  # older than this on first sight: not news any more
RETRY_FOR = pd.Timedelta("6h")  # an item the model never answered for stops being worth a call
SAME_STORY_WINDOW = pd.Timedelta("48h")
STALE_AFTER = pd.Timedelta("1h")  # scoring this far behind means the risk level cannot be trusted
# GDELT finds about 20 new stories a run, which alone would use the daily budget in hours and leave
# official sources waiting. Repeats stay free, so outlets can still confirm a story.
GEO_STORIES_PER_RUN = 3
PRIORITY = {"official": 0, "social": 1, "market": 2, "news": 3, "geo": 4}

TIMES = ["published", "first_seen", "scored_at"]
NUMBERS = ["impact", "confidence"]
TEXTS = ["source", "kind", "title", "url", "cluster", "status", "subject", "stance", "channel",
         "event_type", "assets", "reason", "model", "prompt_version"]
COLUMNS = TIMES + NUMBERS + TEXTS
SCORE_FIELDS = ["impact", "confidence", "subject", "stance", "channel", "event_type", "assets", "reason"]


@dataclass
class NewsRun:
    counts: dict[str, int]
    new: int = 0
    scored: int = 0
    pending: int = 0
    risk: list[dict[str, Any]] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sources_ok(self) -> int:
        return sum(n >= 0 for n in self.counts.values())


def typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed columns and types, so empty and full frames concatenate alike."""
    out = frame.reindex(columns=COLUMNS)
    for col in TIMES:
        out[col] = pd.to_datetime(out[col], utc=True)
    for col in NUMBERS:
        out[col] = pd.to_numeric(out[col])
    for col in TEXTS:
        out[col] = out[col].astype(object).where(out[col].notna(), "")
    return out


def story_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def first_status(item: Item, now: pd.Timestamp) -> str:
    if now - item["published"] > MAX_AGE:
        return OLD
    if not relevant(item):
        return IGNORED
    return SUSPICIOUS if suspicious(item["title"]) else PENDING


def ingest(known: pd.DataFrame, items: list[Item], now: pd.Timestamp) -> pd.DataFrame:
    """New items only, each with a status. A repeat of a recent story joins that story's cluster."""
    seen = set(known["url"])
    recent = known[(known["first_seen"] > now - SAME_STORY_WINDOW) & known["status"].isin([PENDING, SCORED])]
    stories = [(words(t), c) for t, c in zip(recent["title"], recent["cluster"], strict=True)]
    rows = []
    geo_left = GEO_STORIES_PER_RUN
    for item in items:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        status, cluster = first_status(item, now), story_id(item["url"])
        if status == PENDING:
            tokens = words(item["title"])
            match = next((c for other, c in stories if same_story(tokens, other)), None)
            if match is not None:
                status, cluster = DUPLICATE, match
            elif item["kind"] != "geo" or geo_left > 0:
                stories.append((tokens, cluster))
                geo_left -= item["kind"] == "geo"
            else:
                status = IGNORED
        rows.append({**item, "status": status, "cluster": cluster})
    return typed(pd.DataFrame(rows, columns=COLUMNS))


def to_score(frame: pd.DataFrame, now: pd.Timestamp, cfg: NewsConfig) -> pd.DataFrame:
    """What this run may send to the model: official sources first, newest first, within budget."""
    used = int(((frame["status"] == SCORED) & (frame["scored_at"] > now - pd.Timedelta("24h"))).sum())
    room = max(0, min(cfg.max_items_per_run, cfg.max_items_per_day - used))
    pending = frame[frame["status"] == PENDING]
    order = pending.assign(rank=pending["kind"].map(PRIORITY).fillna(9))
    return order.sort_values(["rank", "first_seen"], ascending=[True, False]).head(room)


def apply_scores(frame: pd.DataFrame, answers: dict[int, tuple[Assessment, str]],
                 now: pd.Timestamp) -> None:
    for index, (answer, model) in answers.items():
        values = {**answer.model_dump(include=set(SCORE_FIELDS)), "assets": ",".join(answer.assets),
                  "status": SCORED, "scored_at": now, "model": model, "prompt_version": PROMPT_VERSION}
        for col, value in values.items():
            frame.at[index, col] = value


def share_scores(frame: pd.DataFrame) -> None:
    """Repeats take the score of the story they repeat, so they cost no model call."""
    heads = frame[frame["status"] == SCORED].drop_duplicates("cluster").set_index("cluster")
    waiting = (frame["status"] == DUPLICATE) & frame["impact"].isna()
    open_ = frame[waiting & frame["cluster"].isin(heads.index)]
    for col in [*SCORE_FIELDS, "model", "prompt_version"]:
        frame.loc[open_.index, col] = open_["cluster"].map(heads[col])


def level_changes(previous: pd.DataFrame, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dashboard events for markets whose level differs from the last run."""
    last = previous.sort_values("ts").groupby("scope")["level"].last().to_dict() if len(previous) else {}
    return [{"ts": r["ts"], "type": "news_risk", "message": f"{r['scope']}: {r['level']}", "details": r}
            for r in rows if last.get(r["scope"], "trade") != r["level"]]


def update(scored: pd.DataFrame, risk: pd.DataFrame, sources: dict[str, Source],
           providers: list[Provider], cfg: NewsConfig, now: pd.Timestamp,
           fetch: Fetch = download) -> tuple[pd.DataFrame, pd.DataFrame, NewsRun]:
    """One pass over in-memory tables. Returns the new tables and a report."""
    items, counts = collect(sources, now, fetch)
    fresh = ingest(scored, items, now)
    frame = fresh if scored.empty else pd.concat([scored, fresh]) if len(fresh) else scored
    frame = frame.reset_index(drop=True)
    frame.loc[(frame["status"] == PENDING) & (frame["first_seen"] < now - RETRY_FOR), "status"] = EXPIRED

    batch = to_score(frame, now, cfg)
    payload = [{"id": int(i), "kind": k, "text": t}
               for i, k, t in zip(batch.index, batch["kind"], batch["title"], strict=True)]
    answers = score(providers, payload, now) if payload and providers else {}
    apply_scores(frame, answers, now)
    share_scores(frame)

    pending = frame[frame["status"] == PENDING]
    waiting = pending[pending["kind"] != "geo"]  # the loud source going unread is not a blind spot
    behind = bool((waiting["first_seen"] < now - STALE_AFTER).any())
    report = NewsRun(counts, new=len(fresh), scored=len(answers), pending=len(pending))
    stale = behind or report.sources_ok == 0
    report.risk = assess(frame[frame["impact"].notna()], now, stale)
    report.changes = level_changes(risk, report.risk)
    rows = pd.DataFrame(report.risk)
    return frame, rows if risk.empty else pd.concat([risk, rows], ignore_index=True), report


def _pull(store: Store, path: Path) -> None:
    """Fetch a file. Only a file that does not exist yet may start empty: treating any other
    failure as empty would upload an empty table over the whole history."""
    try:
        path.write_bytes(store.download(BUCKET, path.name))
    except KeyError:  # MemoryStore
        pass
    except Exception as exc:
        if "not found" not in str(exc).lower():
            raise
        log.info("no %s yet, starting empty", path.name)


def run(data_dir: Path, store: Store, sources: dict[str, Source], providers: list[Provider],
        cfg: NewsConfig, now: pd.Timestamp, fetch: Fetch = download) -> NewsRun:
    """Pull both tables, do one pass, push them back, and log level changes."""
    data_dir.mkdir(parents=True, exist_ok=True)
    scored_path, risk_path = data_dir / SCORED_FILE, data_dir / RISK_FILE
    for path in (scored_path, risk_path):
        path.unlink(missing_ok=True)
        _pull(store, path)
    scored = typed(pd.read_parquet(scored_path)) if scored_path.exists() else typed(pd.DataFrame())
    risk = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame()

    scored, risk, report = update(scored, risk, sources, providers, cfg, now, fetch)
    scored.to_parquet(scored_path)
    risk.to_parquet(risk_path)
    for path in (scored_path, risk_path):
        store.upload(BUCKET, path.name, path.read_bytes())
    store.insert("events", report.changes)
    return report
