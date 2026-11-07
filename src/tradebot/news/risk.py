"""Turns scored events into a risk level per market.

A level measures news against the usual: the news stream is almost always somewhat bad, so a level
read against zero cut positions nearly all the time (BTC stood aside 55% of runs, and never traded in
full, over the first two days). Each side's pressure is now how far the sum of its story weights is
above that side's median over the previous days, so an ordinary day is ``trade``.

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
USUAL_HOURS = 72  # the usual is the median of hourly samples over this many hours, where history allows
PRESSURE_SCALE = 1.0  # pressure 1 is one point of story weight above the usual: about two strong stories


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


FRESH = pd.Timedelta("20min")  # one news run: a story first seen in it is new to the bot


def story_weights(events: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    """Signed weight of each story: impact x confidence x source kind, halved every few hours. A
    story repeated by many outlets counts once, at its strongest."""
    if events.empty:
        return pd.Series(dtype=float)
    hours = (now - events["born"]).dt.total_seconds() / 3600
    weight = (events["impact"] * events["confidence"] * events["kind"].map(KIND_WEIGHT).fillna(0.5)
              * 0.5 ** (hours / HALF_LIFE_HOURS))
    strongest = weight.abs().groupby(events["cluster"]).idxmax()
    return pd.Series(weight.loc[strongest].to_numpy(), index=strongest.index)


def news_flow(events: pd.DataFrame, now: pd.Timestamp) -> tuple[float, float]:
    """The balance of recent stories, good minus bad, and the strongest story new this run. Unlike the
    pressures, the balance does not saturate: ten bad stories outweigh one good one."""
    net = float(story_weights(events, now).sum())
    new = events[events["first_seen"] > now - FRESH]
    fresh = story_weights(new.assign(born=new["first_seen"]), now) if len(new) else pd.Series(dtype=float)
    return net, float(fresh.iloc[int(np.argmax(fresh.abs().to_numpy()))]) if len(fresh) else 0.0


def total(events: pd.DataFrame, at: pd.Timestamp) -> float:
    """Sum of signed story weights as the bot knew them at ``at``, from the 24 hours before it."""
    known = events[(events["first_seen"] <= at) & (events["born"] <= at) & (events["born"] > at - WINDOW)]
    return float(story_weights(known, at).sum())


def usual(events: pd.DataFrame, now: pd.Timestamp) -> float:
    """Median of hourly totals over the previous days. Only hours with a full day of history before
    them count; with none, the usual is zero, which is how levels were read before."""
    if events.empty:
        return 0.0
    first = events["first_seen"].min() + WINDOW
    points = [total(events, at) for k in range(1, USUAL_HOURS + 1)
              if (at := now - pd.Timedelta(hours=k)) >= first]
    return float(np.median(points)) if points else 0.0


def above_usual(events: pd.DataFrame, now: pd.Timestamp) -> float:
    """Pressure in [0, 1]: how far this side's weight is above its usual, on ``PRESSURE_SCALE``."""
    weights = events.assign(impact=events["impact"].abs())
    return float(np.clip((total(weights, now) - usual(weights, now)) / PRESSURE_SCALE, 0.0, 1.0))


def assess_scope(scope: str, rated: pd.DataFrame, now: pd.Timestamp, stale: bool) -> dict[str, Any]:
    """``rated`` holds the rated stories of every age the files still keep, for the usual. ``stale``
    only marks that scoring is behind. Stories already rated are still true, so a level stays in
    force: dropping a cut in the middle of a big event would be the worst moment for it."""
    history = rated[rated["assets"].map(lambda a: _covers(a, SCOPES[scope])).astype(bool)]
    mine = history[history["born"] > now - WINDOW]
    bad, good = mine[mine["impact"] < 0], mine[mine["impact"] > 0]
    _, bad_sources = side_pressure(bad, now)
    _, good_sources = side_pressure(good, now)
    negative = above_usual(history[history["impact"] < 0], now)
    positive = above_usual(history[history["impact"] > 0], now)
    long_mult, short_mult = multiplier(negative, bad_sources), multiplier(positive, good_sources)
    net, fresh = news_flow(mine, now)
    net -= usual(history, now)  # the balance against its usual, so a normally gloomy stream reads zero
    worst = min(long_mult, short_mult)
    level = TRADE if worst == 1.0 else STAND_ASIDE if worst == 0.0 else REDUCE
    strongest = mine.reindex(mine["impact"].abs().sort_values(ascending=False).index)
    top = strongest.drop_duplicates("cluster").head(TOP)
    headlines = [f"{i:+.2f} {t[:120]}" for i, t in zip(top["impact"], top["title"], strict=True)]
    return {"ts": now, "scope": scope, "level": level, "long_mult": long_mult, "short_mult": short_mult,
            "negative": negative, "positive": positive, "events": len(mine),
            "sources": int(mine["source"].nunique()), "stale": stale, "net": net, "fresh": fresh,
            "top": " | ".join(headlines)}


def assess(scored: pd.DataFrame, now: pd.Timestamp, stale: bool) -> list[dict[str, Any]]:
    """One row per market, from stories that came out in the last 24 hours.

    ``scored`` holds every row, rated or not: a story's age comes from all of them, including a
    first report that expired before anything rated the story.
    """
    dated = scored.assign(born=story_times(scored))
    # What the user happens to follow must not change the experiment.
    rated = dated[(dated["born"] <= now) & (dated["impact"].abs() >= MIN_IMPACT)
                  & (dated["confidence"] >= MIN_CONFIDENCE) & (dated["kind"] != "watch")]
    return [assess_scope(scope, rated, now, stale) for scope in SCOPES]
