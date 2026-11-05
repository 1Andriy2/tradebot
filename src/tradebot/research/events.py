"""Event studies: does the price behave differently after a kind of event?

Used to test ideas before any strategy is built on them. Hours after an event are compared
with all hours, so a rising market does not look like an effect. Results below ``MIN_EVENTS``
are reported as not enough data instead of being tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradebot.news.risk import MIN_CONFIDENCE, SCOPES
from tradebot.news.scoring import PROMPT_VERSION

MIN_EVENTS = 30
N_SIMS = 2000
STRONG_IMPACT = 0.5


def event_study(close: pd.Series, events: pd.Series, horizon: int, seed: int = 0) -> dict[str, Any]:
    """Forward return after events against the base rate, with a two-sided p-value.

    Events closer together than ``horizon`` count once: overlapping windows are one observation,
    not several. The p-value comes from random event dates, drawn the same number of times.
    """
    forward = (close.shift(-horizon) / close - 1).dropna()
    hits = events.reindex(forward.index, fill_value=False).to_numpy(bool)
    positions: list[int] = []
    for i in np.flatnonzero(hits):
        if not positions or i - positions[-1] >= horizon:
            positions.append(int(i))
    out: dict[str, Any] = {"horizon_hours": horizon, "n": len(positions), "hours": len(forward)}
    if len(positions) < MIN_EVENTS:
        return {**out, "status": "not_enough_data"}
    values = forward.to_numpy()
    observed = float(values[positions].mean())
    base = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (N_SIMS, len(positions)))
    simulated = values[draws].mean(axis=1) - base
    p_value = float((1 + np.sum(np.abs(simulated) >= abs(observed - base))) / (N_SIMS + 1))
    return {**out, "status": "tested", "mean_after": observed, "base_mean": base,
            "p_up_after": float(np.mean(values[positions] > 0)), "base_p_up": float(np.mean(values > 0)),
            "p_value": p_value}


def extreme(series: pd.Series, lookback: int = 24 * 14, quantile: float = 0.95) -> pd.Series:
    """Hours above their own recent ``quantile``, judged only from earlier hours."""
    limit = series.shift(1).rolling(lookback, min_periods=lookback // 2).quantile(quantile)
    return (series > limit) & (series > 0)


def liquidation_studies(close: pd.Series, hourly: pd.DataFrame, stats: pd.DataFrame,
                        horizons: tuple[int, ...] = (4, 24)) -> list[dict[str, Any]]:
    """Does price reverse or continue after liquidation spikes and open-interest jumps?"""
    tests = {
        "long_liquidation_spike": extreme(hourly["long"]) if len(hourly) else None,
        "short_liquidation_spike": extreme(hourly["short"]) if len(hourly) else None,
    }
    if len(stats):
        tests["open_interest_jump"] = extreme(stats["open_interest"].pct_change().abs())
        tests["crowded_longs"] = extreme(stats["long_short_ratio"])
    out = []
    for name, events in tests.items():
        for horizon in horizons:
            study = {"status": "not_enough_data", "n": 0, "horizon_hours": horizon} if events is None \
                else event_study(close, events, horizon)
            out.append({"test": name, **study})
    add_q_values(out)
    return out


def news_studies(close: pd.Series, news: pd.DataFrame, coin: str,
                 horizons: tuple[int, ...] = (4, 24)) -> list[dict[str, Any]]:
    """Does price move after hours with good or bad headlines about ``coin``?"""
    out = []
    about = news[news[coin]] if len(news) else news
    hours = pd.DatetimeIndex(about["published"]).floor("1h") if len(about) else None
    for name, mask in (("good_news", about["tone"] > 0), ("bad_news", about["tone"] < 0)) if len(about) \
            else (("good_news", None), ("bad_news", None)):
        for horizon in horizons:
            if hours is None or mask is None:
                out.append({"test": name, "status": "not_enough_data", "n": 0, "horizon_hours": horizon})
                continue
            events = pd.Series(True, index=hours[mask.to_numpy()].unique())
            out.append({"test": name, **event_study(close, events, horizon)})
    add_q_values(out)
    return out


def _hours(stamps: pd.Series) -> pd.Series:
    """Marks the first candle that closes after each moment. Candles are labelled by their open, so
    a 10:30 event is first usable at the close of the candle labelled 10:00."""
    if stamps.empty:
        return pd.Series(dtype=bool)
    return pd.Series(True, index=pd.DatetimeIndex(stamps).floor("1h").unique())


def scored_news_studies(close: pd.Series, scored: pd.DataFrame, levels: pd.DataFrame, coin: str, model: str,
                        horizons: tuple[int, ...] = (4, 24)) -> list[dict[str, Any]]:
    """Does price move after the model's strong ratings, and after the filter cuts a side?

    Time is ``first_seen``, when the bot really knew. Each story counts once, and only scores from
    the current prompt and ``model`` count, because other scores are not comparable: the fallback
    model rates the same news differently. Levels mix both models, as the bot would have seen them.
    This is the test the filter must pass before it may touch the demo account.
    """
    scope = coin.upper()
    tests: dict[str, pd.Series] = {}
    if len(scored):
        mine = scored[(scored["prompt_version"] == PROMPT_VERSION) & (scored["model"] == model)
                      & (scored["confidence"] >= MIN_CONFIDENCE)
                      & scored["assets"].map(lambda a: bool(SCOPES[scope] & set(a.split(",")))).astype(bool)]
        mine = mine.sort_values("first_seen").drop_duplicates("cluster")
        tests["llm_bad_news"] = _hours(mine.loc[mine["impact"] <= -STRONG_IMPACT, "first_seen"])
        tests["llm_good_news"] = _hours(mine.loc[mine["impact"] >= STRONG_IMPACT, "first_seen"])
    if len(levels):
        mine = levels[(levels["scope"] == scope) & ~levels["stale"].astype(bool)]
        tests["filter_cut_longs"] = _hours(mine.loc[mine["long_mult"] < 1, "ts"])
        tests["filter_cut_shorts"] = _hours(mine.loc[mine["short_mult"] < 1, "ts"])
    out = []
    for name in ("llm_bad_news", "llm_good_news", "filter_cut_longs", "filter_cut_shorts"):
        for horizon in horizons:
            study = event_study(close, tests[name], horizon) if name in tests \
                else {"status": "not_enough_data", "n": 0, "horizon_hours": horizon}
            out.append({"test": name, **study})
    add_q_values(out)
    return out


def add_q_values(rows: list[dict[str, Any]]) -> None:
    """Benjamini-Hochberg across the tested rows of one run."""
    from tradebot.research.patterns import bh_qvalues

    tested = [r for r in rows if r["status"] == "tested"]
    if tested:
        for r, q in zip(tested, bh_qvalues(np.array([r["p_value"] for r in tested])), strict=True):
            r["q_value"] = float(q)
