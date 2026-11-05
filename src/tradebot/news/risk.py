"""Turns scored events into a risk level per market.

Levels can only reduce exposure: bad news cuts longs, good news cuts shorts, and nothing ever
raises a position. Every number here was fixed by reasoning, before any data, and none was tuned.
Only a shadow agent reads these levels (see tradebot.news.gate); the demo account does not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRADE, REDUCE, STAND_ASIDE, UNKNOWN = "trade", "reduce", "stand_aside", "unknown"

SCOPES = {"BTC": {"BTC", "CRYPTO"}, "ETH": {"ETH", "CRYPTO"}, "STOCKS": {"STOCKS"}}
KIND_WEIGHT = {"official": 1.0, "news": 0.8, "market": 0.8, "geo": 0.6, "social": 0.5}
WINDOW = pd.Timedelta("24h")
HALF_LIFE_HOURS = 6.0
MIN_IMPACT = 0.3
MIN_CONFIDENCE = 0.5
REDUCE_AT = 0.4
STAND_ASIDE_AT = 0.75
MIN_SOURCES_TO_STAND_ASIDE = 2  # one voice, even a loud one, can only trigger a reduction
REDUCED = 0.5
TOP = 3


def multiplier(pressure: float, sources: int) -> float:
    if pressure >= STAND_ASIDE_AT and sources >= MIN_SOURCES_TO_STAND_ASIDE:
        return 0.0
    return REDUCED if pressure >= REDUCE_AT else 1.0


def side_pressure(events: pd.DataFrame, now: pd.Timestamp) -> tuple[float, int]:
    """Combined pressure in [0, 1] and the number of distinct sources behind it.

    A story repeated by many outlets counts once at its strongest, and more sources only help
    through the source count. Different stories add up: 1 - product of (1 - weight).
    """
    if events.empty:
        return 0.0, 0
    # News ages from publication, or a day of backlog would look fresh on the run that finds it.
    # A source's date can only make an item older: it is capped by when the bot first saw it.
    hours = (now - events[["published", "first_seen"]].min(axis=1)).dt.total_seconds() / 3600
    weight = (events["impact"].abs() * events["confidence"] * events["kind"].map(KIND_WEIGHT).fillna(0.5)
              * 0.5 ** (hours / HALF_LIFE_HOURS))
    per_story = weight.groupby(events["cluster"]).max()
    return float(1 - np.prod(1 - per_story.clip(0, 1))), int(events["source"].nunique())


def _covers(assets: str, wanted: set[str]) -> bool:
    return bool(wanted & set(assets.split(",")))


def assess_scope(scope: str, events: pd.DataFrame, now: pd.Timestamp, stale: bool) -> dict[str, Any]:
    mine = events[events["assets"].map(lambda a: _covers(a, SCOPES[scope])).astype(bool)]
    bad, good = mine[mine["impact"] < 0], mine[mine["impact"] > 0]
    negative, bad_sources = side_pressure(bad, now)
    positive, good_sources = side_pressure(good, now)
    long_mult, short_mult = multiplier(negative, bad_sources), multiplier(positive, good_sources)
    if stale:  # no judgement without data: behave as if the filter were off, and say so
        long_mult = short_mult = 1.0
    worst = min(long_mult, short_mult)
    level = UNKNOWN if stale else TRADE if worst == 1.0 else STAND_ASIDE if worst == 0.0 else REDUCE
    strongest = mine.reindex(mine["impact"].abs().sort_values(ascending=False).index)
    top = strongest.drop_duplicates("cluster").head(TOP)
    headlines = [f"{i:+.2f} {t[:120]}" for i, t in zip(top["impact"], top["title"], strict=True)]
    return {"ts": now, "scope": scope, "level": level, "long_mult": long_mult, "short_mult": short_mult,
            "negative": negative, "positive": positive, "events": len(mine),
            "sources": int(mine["source"].nunique()), "stale": stale, "top": " | ".join(headlines)}


def assess(scored: pd.DataFrame, now: pd.Timestamp, stale: bool) -> list[dict[str, Any]]:
    """One row per market, from events that were first seen in the last 24 hours."""
    recent = scored[(scored["first_seen"] > now - WINDOW) & (scored["impact"].abs() >= MIN_IMPACT)
                    & (scored["confidence"] >= MIN_CONFIDENCE)]
    return [assess_scope(scope, recent, now, stale) for scope in SCOPES]
