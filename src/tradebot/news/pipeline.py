"""One news run: collect, drop what is known, score within budget, work out the risk level.

Scores are kept as Parquet in the ``data`` bucket. These files are not part of the nightly cache
sync, which would overwrite them with an older copy. ``first_seen`` is when the bot really knew
something, and it is the only honest clock for testing: a model cannot be backtested on old news,
because it may remember what happened next. So items older than a day are stored but never scored.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from tradebot.config import NewsConfig
from tradebot.news.archive import read_file, rotate, write_file
from tradebot.news.filters import relevant, same_story, suspicious, words
from tradebot.news.risk import assess
from tradebot.news.scoring import PROMPT_VERSION, Assessment, Provider, score
from tradebot.news.sources import (
    ASSET_SOURCES,
    BACKFILL_DAYS,
    Fetch,
    Item,
    Source,
    collect,
    download,
    needs_backfill,
)
from tradebot.store import Store

log = logging.getLogger(__name__)

SCORED_STEM, RISK_STEM = "news_scored", "news_risk"
SCORED_FILE, RISK_FILE = f"{SCORED_STEM}.parquet", f"{RISK_STEM}.parquet"

PENDING, SCORED, DUPLICATE, IGNORED, OLD, EXPIRED, SUSPICIOUS, OFF_TOPIC = (
    "pending", "scored", "duplicate", "ignored", "old", "expired", "suspicious", "off_topic")

MAX_AGE = pd.Timedelta("24h")  # older than this on first sight: not news any more
RETRY_FOR = pd.Timedelta("6h")  # an item the model never answered for stops being worth a call
SAME_STORY_WINDOW = pd.Timedelta("48h")
# What a run must remember. The rest costs nothing to judge again, and keeping it made the file that is
# fetched 3000 times a month several times larger: measured, 26 GB a month against 5 GB allowed.
REMEMBERED = [PENDING, SCORED, DUPLICATE, EXPIRED, OFF_TOPIC]
STALE_AFTER = pd.Timedelta("1h")  # scoring this far behind means the risk level cannot be trusted
STALE_BACKLOG = 5  # a few items the model will not answer must not switch the whole filter off
REPUBLISH = pd.Timedelta("6h")  # a failed write to the page's tables is repaired within this time
# GDELT finds about 20 new stories a run, which alone would use the daily budget in hours and leave
# official sources waiting. Repeats stay free, so outlets can still confirm a story.
GEO_STORIES_PER_RUN = 3
# Official items come first: there are few, and the risk filter depends on them. Followed assets next.
PRIORITY = {"official": 0, "watch": 1, "social": 2, "market": 3, "news": 4, "geo": 5}
HISTORY_RANK = 9  # a new subscription's old stories wait behind everything that happened today

TIMES = ["published", "first_seen", "scored_at"]
NUMBERS = ["impact", "confidence"]
TEXTS = ["source", "kind", "title", "title_uk", "url", "cluster", "status", "subject", "stance", "channel",
         "event_type", "assets", "reason", "model", "prompt_version", "watch", "ticker"]
COLUMNS = TIMES + NUMBERS + TEXTS
SCORE_FIELDS = ["impact", "confidence", "subject", "stance", "channel", "event_type", "assets", "title_uk",
                "reason"]
PAGE_FIELDS = ["cluster", "first_seen", "published", "source", "kind", "title", "title_uk", "url", "subject",
               "stance", "impact", "confidence", "channel", "event_type", "assets", "reason"]
LEVEL_FIELDS = ["scope", "level", "long_mult", "short_mult", "negative", "positive", "events", "sources",
                "stale"]


@dataclass
class NewsRun:
    counts: dict[str, int]
    new: int = 0
    scored: int = 0
    pending: int = 0
    risk: list[dict[str, Any]] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)

    @property
    def sources_ok(self) -> int:
        return sum(n >= 0 for n in self.counts.values())

    @property
    def general_ok(self) -> int:
        """General sources that answered. Followed assets must not make the filter look healthy."""
        return sum(n >= 0 for name, n in self.counts.items() if not name.startswith(ASSET_SOURCES))


def typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed columns and types, so empty and full frames concatenate alike."""
    out = frame.reindex(columns=COLUMNS)
    for col in TIMES:
        # A column of only empty values would otherwise get a unit too coarse to hold real clocks.
        out[col] = pd.to_datetime(out[col], utc=True).astype("datetime64[us, UTC]")
    for col in NUMBERS:
        out[col] = pd.to_numeric(out[col])
    for col in TEXTS:
        out[col] = out[col].astype(object).where(out[col].notna(), "")
    return out


def story_id(url: str, lane: str = "") -> str:
    return hashlib.sha256(f"{lane}{url}".encode()).hexdigest()[:16]


def first_status(item: Item, now: pd.Timestamp) -> str:
    limit = pd.Timedelta(days=BACKFILL_DAYS) if item.get("backfill") else MAX_AGE
    if now - item["published"] > limit:
        return OLD
    if not relevant(item):
        return IGNORED
    return SUSPICIOUS if suspicious(item["title"]) else PENDING


def ingest(known: pd.DataFrame, items: list[Item], now: pd.Timestamp) -> pd.DataFrame:
    """New items only, each with a status. A repeat of a recent story joins that story's cluster.

    The general stream and each followed asset are separate lanes that share nothing, not even a
    link. So following an asset cannot change what the general stream, and with it the risk filter,
    sees. The price is that a story in two lanes is rated twice.
    """
    seen = set(zip(known["watch"], known["url"], strict=True))
    recent = known[(known["first_seen"] > now - SAME_STORY_WINDOW)
                   & known["status"].isin([PENDING, SCORED, EXPIRED, OFF_TOPIC])]
    stories: dict[str, list[tuple[frozenset[str], str]]] = {}
    for lane, title, story in zip(recent["watch"], recent["title"], recent["cluster"], strict=True):
        stories.setdefault(lane, []).append((words(title), story))
    rows = []
    geo_left = GEO_STORIES_PER_RUN
    for item in items:
        lane = item.get("watch", "")
        if (lane, item["url"]) in seen:
            continue
        seen.add((lane, item["url"]))
        status, cluster = first_status(item, now), story_id(item["url"], lane)
        if status == PENDING:
            tokens = words(item["title"])
            # Market items and filings share a template, so their words say nothing.
            templated = item["kind"] == "market" or item["source"] == "sec_8k"
            match = None if templated else next(
                (c for other, c in stories.get(lane, []) if same_story(tokens, other)), None)
            if match is not None:
                status, cluster = DUPLICATE, match
            elif item["kind"] != "geo" or geo_left > 0:
                stories.setdefault(lane, []).append((tokens, cluster))
                geo_left -= item["kind"] == "geo"
            else:
                status = IGNORED
        rows.append({**item, "status": status, "cluster": cluster})
    return typed(pd.DataFrame(rows, columns=COLUMNS))


def choose_heads(frame: pd.DataFrame) -> None:
    """Each unrated story waits on its best item: the highest-priority source, then the oldest. So
    an official repeat is not held behind a GDELT item, and a story outlives an expired first item."""
    rated = frame.loc[frame["status"].isin([SCORED, OFF_TOPIC]), "cluster"]
    waiting = frame[frame["status"].isin([PENDING, DUPLICATE]) & frame["impact"].isna()
                    & ~frame["cluster"].isin(rated)]
    if waiting.empty:
        return
    ranked = waiting.assign(rank=waiting["kind"].map(PRIORITY).fillna(9)).sort_values(
        ["rank", "first_seen"], kind="stable")
    heads = ranked.drop_duplicates("cluster").index
    frame.loc[waiting.index, "status"] = DUPLICATE
    frame.loc[heads, "status"] = PENDING


def to_score(frame: pd.DataFrame, now: pd.Timestamp, cfg: NewsConfig) -> pd.DataFrame:
    """What this run may send to the model: official sources first, newest first, within budget."""
    asked = frame["status"].isin([SCORED, OFF_TOPIC])
    used = int((asked & (frame["scored_at"] > now - pd.Timedelta("24h"))).sum())
    room = max(0, min(cfg.max_items_per_run, cfg.max_items_per_day - used))
    pending = frame[frame["status"] == PENDING]
    order = pending.assign(rank=pending["kind"].map(PRIORITY).fillna(9))
    order.loc[order["first_seen"] - order["published"] > MAX_AGE, "rank"] = HISTORY_RANK
    return order.sort_values(["rank", "first_seen", "published"], ascending=[True, False, False]).head(room)


def apply_scores(frame: pd.DataFrame, answers: dict[int, tuple[Assessment, str]],
                 done: pd.Timestamp) -> None:
    for index, (answer, model) in answers.items():
        if frame.at[index, "watch"] and not answer.about_asset:
            frame.at[index, "status"], frame.at[index, "scored_at"] = OFF_TOPIC, done
            continue
        values = {**answer.model_dump(include=set(SCORE_FIELDS)), "assets": ",".join(answer.assets),
                  "status": SCORED, "scored_at": done, "model": model, "prompt_version": PROMPT_VERSION}
        for col, value in values.items():
            frame.at[index, col] = value


def share_scores(frame: pd.DataFrame) -> None:
    """Repeats take the score of the story they repeat, so they cost no model call."""
    heads = frame[frame["status"] == SCORED].drop_duplicates("cluster").set_index("cluster")
    waiting = (frame["status"] == DUPLICATE) & frame["impact"].isna()
    open_ = frame[waiting & frame["cluster"].isin(heads.index)]
    for col in [*SCORE_FIELDS, "model", "prompt_version"]:
        frame.loc[open_.index, col] = open_["cluster"].map(heads[col])


def page_rows(frame: pd.DataFrame, clusters: set[str] | None = None) -> list[dict[str, Any]]:
    """One row per story for the dashboard, with every outlet that told it."""
    rated = frame[frame["impact"].notna() & frame["status"].isin([SCORED, DUPLICATE])]
    if clusters is not None:
        rated = rated[rated["cluster"].isin(clusters)]
    outlets = rated.groupby("cluster")["source"].unique()
    tickers = rated[rated["ticker"] != ""].groupby("cluster")["ticker"].unique().to_dict()
    heads = rated[rated["status"] == SCORED].drop_duplicates("cluster")
    rows = cast(list[dict[str, Any]], heads[PAGE_FIELDS].to_dict("records"))
    return [{**row, "outlets": [str(s) for s in outlets[row["cluster"]]],
             "assets": ",".join(dict.fromkeys([*tickers.get(row["cluster"], []), *row["assets"].split(",")]))
             .strip(",")} for row in rows]


def rated_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """What is worth archiving. The rest was only kept to recognise a link already seen."""
    return frame[frame["status"].isin([SCORED, DUPLICATE])]


def publish(store: Store, frame: pd.DataFrame, report: NewsRun, now: pd.Timestamp) -> None:
    """Update the page's tables. The Parquet files are the record, so a failure here stops nothing.

    Recent stories go again each run, so a write that failed once does not lose them for good.
    """
    recent = set(frame.loc[frame["scored_at"] > now - REPUBLISH, "cluster"])
    levels = [{**{k: r[k] for k in LEVEL_FIELDS}, "updated_at": r["ts"]} for r in report.risk]
    for table, rows, key in (("news_items", page_rows(frame, report.touched | recent), "cluster"),
                             ("news_levels", levels, "scope")):
        try:
            store.upsert(table, rows, on_conflict=key)
        except Exception as exc:
            log.warning("%s not updated: %s", table, type(exc).__name__)


def level_changes(previous: pd.DataFrame, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dashboard events for markets whose level, or the side it cuts, differs from the last run."""
    calm = ("trade", 1.0, 1.0)
    last: dict[str, tuple[str, float, float]] = {}
    if len(previous):
        latest = previous.sort_values("ts").groupby("scope").last()
        last = {str(s): (r["level"], r["long_mult"], r["short_mult"]) for s, r in latest.iterrows()}
    return [{"ts": r["ts"], "type": "news_risk", "message": f"{r['scope']}: {r['level']}", "details": r}
            for r in rows if last.get(r["scope"], calm) != (r["level"], r["long_mult"], r["short_mult"])]


def update(scored: pd.DataFrame, risk: pd.DataFrame, sources: dict[str, Source],
           providers: list[Provider], cfg: NewsConfig, now: pd.Timestamp,
           fetch: Fetch = download, clock: Callable[[], pd.Timestamp] | None = None,
           watched: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, NewsRun]:
    """One pass over in-memory tables. Returns the new tables and a report.

    ``now`` is the start of the run. Ratings and levels are stamped by ``clock`` once they exist,
    because tests of the filter must not credit it with a moment before it knew. ``watched`` maps
    the ids of followed assets to their names, and anything unsubscribed stops at once. ``None``
    means the list could not be read, which is not the same as nothing being followed.
    """
    items, counts = collect(sources, now, fetch)
    fresh = ingest(scored, items, now)
    frame = fresh if scored.empty else pd.concat([scored, fresh]) if len(fresh) else scored
    frame = frame.reset_index(drop=True)
    frame.loc[(frame["status"] == PENDING) & (frame["first_seen"] < now - RETRY_FOR), "status"] = EXPIRED
    if watched is not None:
        unrated = frame["status"].isin([PENDING, DUPLICATE]) & frame["impact"].isna()
        frame.loc[unrated & (frame["watch"] != "") & ~frame["watch"].isin(watched), "status"] = IGNORED
    choose_heads(frame)

    batch = to_score(frame, now, cfg)
    names = watched or {}
    batch = batch[(batch["watch"] == "") | batch["watch"].isin(names)]  # the rest wait for their list
    payload = [{"id": int(i), "kind": r["kind"], "text": r["title"],
                **({"asset": f"{names[r['watch']]} ({r['ticker']})"} if r["watch"] else {})}
               for i, r in zip(batch.index, batch.to_dict("records"), strict=True)]
    answers = score(providers, payload, now) if payload and providers else {}
    done = clock() if clock else now
    apply_scores(frame, answers, done)
    share_scores(frame)

    pending = frame[frame["status"] == PENDING]
    waiting = pending[~pending["kind"].isin(["geo", "watch"])]  # neither of these blinds the filter
    behind = int((waiting["first_seen"] < now - STALE_AFTER).sum()) >= STALE_BACKLOG
    touched = set(frame.loc[list(answers), "cluster"]) | set(fresh["cluster"])
    report = NewsRun(counts, new=len(fresh), scored=len(answers), pending=len(pending), touched=touched)
    stale = behind or report.general_ok == 0
    report.risk = assess(frame, done, stale)
    report.changes = level_changes(risk, report.risk)
    rows = pd.DataFrame(report.risk).drop(columns="top")  # headlines go to events, not this file
    return frame, rows if risk.empty else pd.concat([risk, rows], ignore_index=True), report


def mark_backfilled(store: Store, watchlist: list[Item], counts: dict[str, int], now: pd.Timestamp) -> None:
    """Close the history window of assets whose search answered. A failed search tries again next run,
    and stories already found wait in the file, so no model outage can lose them. Updating, rather than
    upserting, cannot bring back an asset that was unsubscribed during the run."""
    for asset in watchlist:
        if needs_backfill(asset) and counts.get(f"google_{asset['id']}", -1) >= 0:
            store.update("watchlist", {"backfilled_at": now}, {"id": asset["id"]})


def run(store: Store, sources: dict[str, Source], providers: list[Provider], cfg: NewsConfig,
        now: pd.Timestamp, fetch: Fetch = download, clock: Callable[[], pd.Timestamp] | None = None,
        watched: dict[str, str] | None = None) -> NewsRun:
    """Read both tables, do one pass, write them back, and log level changes.

    Only a file that does not exist yet starts empty. Any other failure to read stops the run:
    going on would upload an empty table over the whole history.
    """
    known = read_file(store, SCORED_FILE)
    scored = typed(pd.DataFrame() if known is None else known)
    risk = read_file(store, RISK_FILE)
    risk = pd.DataFrame() if risk is None else risk

    scored, risk, report = update(scored, risk, sources, providers, cfg, now, fetch, clock, watched)
    publish(store, scored, report, now)
    scored = scored[scored["status"].isin(REMEMBERED)]
    scored = rotate(store, SCORED_STEM, scored, "first_seen", now, rated_rows, together="cluster")
    risk = rotate(store, RISK_STEM, risk, "ts", now)
    write_file(store, SCORED_FILE, scored)
    write_file(store, RISK_FILE, risk)
    store.insert("events", report.changes)
    return report
