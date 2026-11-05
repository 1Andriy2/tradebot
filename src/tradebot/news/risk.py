"""Turns scored events into a risk level per market.

Levels can only reduce exposure: bad news cuts longs, good news cuts shorts, and nothing ever
raises a position. Every number here was fixed by reasoning, before any data, and none was tuned.
Only a shadow agent reads these levels (see tradebot.news.gate); the demo account does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADE, REDUCE, STAND_ASIDE = "trade", "reduce", "stand_aside"

SCOPES = {"BTC": {"BTC", "CRYPTO"}, "ETH": {"ETH", "CRYPTO"}, "STOCKS": {"STOCKS"}}
KIND_WEIGHT = {"official": 1.0, "news": 0.8, "watch": 0.8, "market": 0.8, "geo": 0.6, "social": 0.5}
WINDOW = pd.Timedelta("24h")
HALF_LIFE_HOURS = 6.0
MIN_IMPACT = 0.3
MIN_CONFIDENCE = 0.5
REDUCE_AT = 0.4
STAND_ASIDE_AT = 0.75
MIN_SOURCES_TO_STAND_ASIDE = 2  # one voice, even a loud one, can only trigger a reduction
MIN_SOURCE_WEIGHT = 0.2  # a faint or fading item does not count as a second voice
REDUCED = 0.5
TOP = 3


def multiplier(pressure: float, sources: int) -> float:
    if pressure >= STAND_ASIDE_AT and sources >= MIN_SOURCES_TO_STAND_ASIDE:
        return 0.0
    return REDUCED if pressure >= REDUCE_AT else 1.0


def story_times(events: pd.DataFrame) -> pd.Series:
    """When each row's story first came out, so a repeat cannot make an old story young again."""
    earliest = events[["published", "first_seen"]].min(axis=1)
    return earliest.groupby(events["cluster"]).transform("min")


def side_pressure(events: pd.DataFrame, now: pd.Timestamp) -> tuple[float, int]:
    """Combined pressure in [0, 1] and the number of distinct sources behind it.

    A story repeated by many outlets counts once at its strongest, and more sources only help
    through the source count. Different stories add up: 1 - product of (1 - weight).
    """
    if events.empty:
        return 0.0, 0
    # News ages from publication, or a day of backlog would look fresh on the run that finds it.
    # A source's date can only make an item older: it is capped by when the bot first saw it.
    hours = (now - events["born"]).dt.total_seconds() / 3600
    weight = (events["impact"].abs() * events["confidence"] * events["kind"].map(KIND_WEIGHT).fillna(0.5)
              * 0.5 ** (hours / HALF_LIFE_HOURS))
    per_story = weight.groupby(events["cluster"]).max()
    sources = events.loc[weight >= MIN_SOURCE_WEIGHT, "source"].nunique()
    return float(1 - np.prod(1 - per_story.clip(0, 1))), int(sources)


def _covers(assets: str, wanted: set[str]) -> bool:
    return bool(wanted & set(assets.split(",")))


def assess_scope(scope: str, events: pd.DataFrame, now: pd.Timestamp, stale: bool) -> dict[str, Any]:
    """``stale`` only marks that scoring is behind. Stories already rated are still true, so a level
    stays in force: dropping a cut in the middle of a big event would be the worst moment for it."""
    mine = events[events["assets"].map(lambda a: _covers(a, SCOPES[scope])).astype(bool)]
    bad, good = mine[mine["impact"] < 0], mine[mine["impact"] > 0]
    negative, bad_sources = side_pressure(bad, now)
    positive, good_sources = side_pressure(good, now)
    long_mult, short_mult = multiplier(negative, bad_sources), multiplier(positive, good_sources)
    worst = min(long_mult, short_mult)
    level = TRADE if worst == 1.0 else STAND_ASIDE if worst == 0.0 else REDUCE
    strongest = mine.reindex(mine["impact"].abs().sort_values(ascending=False).index)
    top = strongest.drop_duplicates("cluster").head(TOP)
    headlines = [f"{i:+.2f} {t[:120]}" for i, t in zip(top["impact"], top["title"], strict=True)]
    return {"ts": now, "scope": scope, "level": level, "long_mult": long_mult, "short_mult": short_mult,
            "negative": negative, "positive": positive, "events": len(mine),
            "sources": int(mine["source"].nunique()), "stale": stale, "top": " | ".join(headlines)}


def assess(scored: pd.DataFrame, now: pd.Timestamp, stale: bool) -> list[dict[str, Any]]:
    """One row per market, from stories that came out in the last 24 hours.

    ``scored`` holds every row, rated or not: a story's age comes from all of them, including a
    first report that expired before anything rated the story.
    """
    dated = scored.assign(born=story_times(scored))
    # What the user happens to follow must not change the experiment.
    recent = dated[(dated["born"] > now - WINDOW) & (dated["impact"].abs() >= MIN_IMPACT)
                   & (dated["confidence"] >= MIN_CONFIDENCE) & (dated["kind"] != "watch")]
    return [assess_scope(scope, recent, now, stale) for scope in SCOPES]
