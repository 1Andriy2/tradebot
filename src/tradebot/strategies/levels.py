"""Levels the price seldom reaches.

For each bar, the share of the last ``window`` bars whose high-low range touched the price band the
close is in. A band touched by under ``rare`` of them is a level price seldom visits. Above the
window's median close it is a high the market has rarely accepted, so the strategy sells it; below
the median, it buys. It exits when price is back where it trades often. The bands are steps of
``band`` in log price, so they are the same width in percent at any price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.strategies.base import Strategy


def visit_share(df: pd.DataFrame, window: int, band: float) -> pd.Series:
    """Share of the previous ``window`` bars whose range touched each bar's closing band."""
    low = np.floor(np.log(df["low"].to_numpy(float)) / band).astype(np.int64)
    high = np.floor(np.log(df["high"].to_numpy(float)) / band).astype(np.int64)
    close = np.floor(np.log(df["close"].to_numpy(float)) / band).astype(np.int64)
    base = int(min(low.min(), close.min()))
    counts = np.zeros(int(max(high.max(), close.max())) - base + 1, dtype=np.int64)
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        if i >= window:
            out[i] = counts[close[i] - base] / window
            old = i - window
            counts[low[old] - base: high[old] - base + 1] -= 1
        counts[low[i] - base: high[i] - base + 1] += 1
    return pd.Series(out, index=df.index)


class RareLevel(Strategy):
    name = "rare_level"
    history_bars = 6400  # shares the long fetch of the 4-hour strategy; covers a 30-day window on 15m

    def __init__(self, window: int = 2880, band: float = 0.005, rare: float = 0.02,
                 usual: float = 0.10) -> None:
        super().__init__(window=window, band=band, rare=rare, usual=usual)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        share = visit_share(df, p["window"], p["band"]).to_numpy()
        middle = df["close"].rolling(p["window"]).median().shift(1).to_numpy()
        close = df["close"].to_numpy(float)
        out = np.full(len(df), np.nan)
        pos = 0.0
        for i in range(len(df)):
            if np.isnan(share[i]) or np.isnan(middle[i]):
                continue
            if pos != 0.0 and share[i] >= p["usual"]:
                pos = 0.0
            if share[i] < p["rare"]:
                pos = -1.0 if close[i] > middle[i] else 1.0
            out[i] = pos
        return pd.Series(out, index=df.index)
