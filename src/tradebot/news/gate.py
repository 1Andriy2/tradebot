"""Turns news risk levels into a position scale, for the shadow agents that test the filter.

A bar sees only levels written before it closed, so this cannot look ahead. Multipliers are cut
to the 0..1 range here as well, so even a damaged file can only reduce exposure. A level older than
``MAX_AGE`` is ignored: after an outage the agent falls back to its plain signal.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from tradebot.news.archive import BUCKET
from tradebot.news.pipeline import RISK_FILE
from tradebot.store import Store

MAX_AGE = pd.Timedelta("1h")


def _utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """One time unit on both sides, which ``merge_asof`` requires."""
    return pd.Series(pd.to_datetime(values, utc=True)).astype("datetime64[us, UTC]")


def load_levels(store: Store) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.download(BUCKET, RISK_FILE)))


def exposure_scale(signal: pd.Series, levels: pd.DataFrame, scope: str, bar: pd.Timedelta) -> pd.Series:
    """Share of each bar's position to take: the multiplier in force when the bar closed, for the
    side the signal is on."""
    full = pd.Series(1.0, index=signal.index)
    if levels.empty:
        return full
    mine = levels[levels["scope"] == scope].sort_values("ts")
    if mine.empty:
        return full
    mine = mine.assign(ts=_utc(mine["ts"]).to_numpy())
    closes = pd.DataFrame({"close": _utc(pd.DatetimeIndex(signal.index) + bar)})
    merged = pd.merge_asof(closes, mine[["ts", "long_mult", "short_mult"]], left_on="close", right_on="ts",
                           direction="backward", tolerance=MAX_AGE)
    long_mult = merged["long_mult"].fillna(1.0).clip(0, 1).to_numpy()
    short_mult = merged["short_mult"].fillna(1.0).clip(0, 1).to_numpy()
    return pd.Series(np.where(signal.to_numpy() < 0, short_mult, long_mult), index=signal.index)
