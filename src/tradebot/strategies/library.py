"""Strategy library for research: rule-based families plus pattern strategies fitted on
training data only. Each family has a parameter grid explored by walk-forward research."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tradebot import indicators as ind
from tradebot.strategies.base import Strategy, hysteresis
from tradebot.strategies.baselines import (
    BollingerBreakout,
    BuyHold,
    Grid,
    MACrossover,
    RSIMeanReversion,
)
from tradebot.strategies.regimes import ChandelierTrend, RegimeSwitch, VolScaledTrend


class DonchianBreakout(Strategy):
    """Turtle-style: enter on a break of the prior n-bar high/low, exit on the n/2 channel."""

    name = "donchian"

    def __init__(self, n: int = 55) -> None:
        super().__init__(n=n)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        n = self.params["n"]
        hi = df["high"].rolling(n, min_periods=n).max().shift()
        lo = df["low"].rolling(n, min_periods=n).min().shift()
        m = max(2, n // 2)
        exit_hi = df["high"].rolling(m, min_periods=m).max().shift()
        exit_lo = df["low"].rolling(m, min_periods=m).min().shift()
        c = df["close"]
        sig = hysteresis(c > hi, c < exit_lo, c < lo, c > exit_hi)
        return sig.where(hi.notna())


class MACDTrend(Strategy):
    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        super().__init__(fast=fast, slow=slow, signal=signal)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        line, sig = ind.macd(df["close"], self.params["fast"], self.params["slow"], self.params["signal"])
        return pd.Series(np.sign(line - sig), index=df.index)


class KeltnerBreakout(Strategy):
    """Break of EMA ± k·ATR; exit when price returns to the EMA."""

    name = "keltner"

    def __init__(self, n: int = 20, k: float = 2.0) -> None:
        super().__init__(n=n, k=k)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        n, k = self.params["n"], self.params["k"]
        mid = ind.ema(df["close"], n)
        width = k * ind.atr(df, n)
        c = df["close"]
        sig = hysteresis(c > mid + width, c < mid, c < mid - width, c > mid)
        return sig.where(width.notna())


class TimeSeriesMomentum(Strategy):
    """Sign of the trailing n-bar return."""

    name = "tsmom"

    def __init__(self, n: int = 96) -> None:
        super().__init__(n=n)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        ret = df["close"] / df["close"].shift(self.params["n"]) - 1
        return pd.Series(np.sign(ret), index=df.index)


class ZScoreReversion(Strategy):
    """Fade |z| > z_in of price vs its rolling mean; exit when |z| < z_out."""

    name = "zscore"

    def __init__(self, n: int = 48, z_in: float = 2.0, z_out: float = 0.5) -> None:
        super().__init__(n=n, z_in=z_in, z_out=z_out)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        c = df["close"]
        z = (c - ind.sma(c, p["n"])) / c.rolling(p["n"], min_periods=p["n"]).std()
        sig = hysteresis(z < -p["z_in"], z > -p["z_out"], z > p["z_in"], z < p["z_out"])
        return sig.where(z.notna())


class ADXTrend(Strategy):
    """EMA-cross direction, only while ADX says the market is trending."""

    name = "adx_trend"

    def __init__(self, fast: int = 20, slow: int = 100, adx_min: float = 25.0) -> None:
        super().__init__(fast=fast, slow=slow, adx_min=adx_min)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        trend = np.sign(ind.ema(df["close"], p["fast"]) - ind.ema(df["close"], p["slow"]))
        strength = ind.adx(df, 14)
        out = pd.Series(np.where(strength > p["adx_min"], trend, 0.0), index=df.index)
        return out.where(strength.notna())


class Seasonality(Strategy):
    """Pattern strategy: trade the hours of day whose average next-bar return was
    significantly positive/negative in the training data (|t| > t_min)."""

    name = "seasonality"
    uses_stops = False

    def __init__(self, t_min: float = 2.0) -> None:
        super().__init__(t_min=t_min)
        self.side_by_hour: dict[int, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        nxt = df["close"].pct_change().shift(-1)
        hours = pd.DatetimeIndex(df.index).hour
        stats = nxt.groupby(hours).agg(["mean", "std", "count"])
        t = stats["mean"] / (stats["std"] / np.sqrt(stats["count"]))
        self.side_by_hour = {int(h): float(np.sign(v)) for h, v in t.items() if abs(v) > self.params["t_min"]}

    def generate(self, df: pd.DataFrame) -> pd.Series:
        hours = pd.DatetimeIndex(df.index).hour
        return pd.Series([self.side_by_hour.get(int(h), 0.0) for h in hours], index=df.index)


# family -> (factory, parameter grid)
ParamGrid = list[dict[str, Any]]
FAMILIES: dict[str, tuple[Callable[..., Strategy], ParamGrid]] = {
    "buy_hold": (BuyHold, [{}]),
    "ma_cross": (MACrossover, [{"fast": f, "slow": s} for f in (10, 20, 50) for s in (100, 200) if f < s]),
    "rsi_mr": (RSIMeanReversion, [{"period": p, "low": lo, "high": 100 - lo}
                                  for p in (7, 14) for lo in (20, 30)]),
    "bb_breakout": (BollingerBreakout, [{"period": p, "k": k} for p in (20, 50) for k in (2.0, 2.5)]),
    "grid": (Grid, [{"anchor": a, "step": st} for a in (96, 288) for st in (0.01, 0.02)]),
    "donchian": (DonchianBreakout, [{"n": n} for n in (20, 55, 100, 200)]),
    "macd": (MACDTrend, [{"fast": f, "slow": s, "signal": 9} for f, s in ((12, 26), (24, 52), (48, 104))]),
    "keltner": (KeltnerBreakout, [{"n": n, "k": k} for n in (20, 50) for k in (1.5, 2.5)]),
    "tsmom": (TimeSeriesMomentum, [{"n": n} for n in (24, 96, 288, 672)]),
    "zscore": (ZScoreReversion, [{"n": n, "z_in": z} for n in (48, 192) for z in (2.0, 2.5)]),
    "adx_trend": (ADXTrend, [{"fast": f, "slow": s, "adx_min": a}
                             for f, s in ((20, 100), (50, 200)) for a in (20.0, 30.0)]),
    "seasonality": (Seasonality, [{"t_min": t} for t in (2.0, 3.0)]),
    "regime_switch": (RegimeSwitch, [{"adx_min": a, "lookback": n}
                                     for a in (20.0, 25.0) for n in (100, 500)]),
    "vol_scaled_trend": (VolScaledTrend, [{"fast": f, "slow": s, "lookback": n}
                                          for f, s in ((20, 100), (50, 200)) for n in (100, 500)]),
    "chandelier_trend": (ChandelierTrend, [{"mult": m, "max_hold": h}
                                           for m in (2.5, 3.0) for h in (96, 288)]),
}


def make(family: str, params: dict[str, Any]) -> Strategy:
    factory, _ = FAMILIES[family]
    return factory(**params)
