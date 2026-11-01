"""Baseline strategies every ML/RL agent is compared against."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot import indicators as ind
from tradebot.strategies.base import Strategy, hysteresis


class BuyHold(Strategy):
    name = "buy_hold"
    uses_stops = False

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


class MACrossover(Strategy):
    """Long when the fast EMA is above the slow EMA, short when below."""

    name = "ma_cross"

    def __init__(self, fast: int = 20, slow: int = 100) -> None:
        if fast >= slow:
            raise ValueError("fast must be < slow")
        super().__init__(fast=fast, slow=slow)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        fast = ind.ema(df["close"], self.params["fast"])
        slow = ind.ema(df["close"], self.params["slow"])
        return pd.Series(np.sign(fast - slow), index=df.index)


class RSIMeanReversion(Strategy):
    """Long below `low`, short above `high`; exit when RSI returns to the middle."""

    name = "rsi_mr"

    def __init__(self, period: int = 14, low: float = 30, high: float = 70, exit: float = 50) -> None:
        super().__init__(period=period, low=low, high=high, exit=exit)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        r = ind.rsi(df["close"], p["period"])
        sig = hysteresis(r < p["low"], r >= p["exit"], r > p["high"], r <= p["exit"])
        return sig.where(r.notna())


class BollingerBreakout(Strategy):
    """Long on a close above the upper band, short below the lower; exit at the middle band."""

    name = "bb_breakout"

    def __init__(self, period: int = 20, k: float = 2.0) -> None:
        super().__init__(period=period, k=k)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        lower, mid, upper = ind.bollinger(close, self.params["period"], self.params["k"])
        sig = hysteresis(close > upper, close < mid, close < lower, close > mid)
        return sig.where(mid.notna())


class Grid(Strategy):
    """Futures grid around a moving anchor: sell into rallies, buy into dips, one level per step."""

    name = "grid"

    def __init__(self, anchor: int = 96, step: float = 0.01, levels: int = 4) -> None:
        super().__init__(anchor=anchor, step=step, levels=levels)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        anchor = ind.sma(df["close"], p["anchor"])
        dev = (df["close"] - anchor) / anchor
        level = np.trunc(dev / p["step"]).clip(-p["levels"], p["levels"])
        return pd.Series(-level / p["levels"], index=df.index)


def baseline_strategies() -> list[Strategy]:
    return [BuyHold(), MACrossover(), RSIMeanReversion(), BollingerBreakout(), Grid()]
