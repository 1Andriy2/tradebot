"""Daily trend: long while the last ``n`` days rose, flat otherwise (or short, if ``long_only`` is off).

Chosen from five candidates fixed in advance (daily momentum long/short and long-only, a 20-day
breakout, funding-rate contrarian, a fade of 1-hour shocks) by one rule set before the test: a gain on
BTC and on ETH from 2022 to 2026-11, and in each half, with fees, slippage and funding. Only the
long-only version passed; 10 of 12 lookbacks from 14 to 84 days gained too. One pass in five can
still be luck, which is why it starts on paper.

It decides on completed daily candles only, so it trades at most once a day, and it has no ATR stop:
a stop sized on 15-minute candles would cut a daily trend on noise. The account's kill switch stays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.ml.features import resample_ohlcv
from tradebot.strategies.base import Strategy

DAY = pd.Timedelta("1D")


class DailyTrend(Strategy):
    name = "trend_1d"
    uses_stops = False
    history_bars = 6400  # 66 days of 15m candles: shares the long fetch of the other slow strategies

    def __init__(self, n: int = 28, long_only: bool = True) -> None:
        super().__init__(n=n, long_only=long_only)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        base = pd.Timedelta(pd.Series(df.index).diff().median())
        days = resample_ohlcv(df, "1D")
        ret = days["close"] / days["close"].shift(self.params["n"]) - 1
        side = np.sign(ret).clip(lower=0) if self.params["long_only"] else np.sign(ret)
        # A day labelled d is complete at the last bar inside it, and its signal holds until the next.
        done = pd.DatetimeIndex(days.index) + (DAY - base)
        return pd.Series(side.to_numpy(), index=done).reindex(df.index, method="ffill")
