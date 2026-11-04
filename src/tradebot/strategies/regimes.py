"""Strategies that adapt to the market regime or manage their own exits.

Ideas and default values come from the regime-detection and exit-strategies skills in
agiprolabs/claude-trading-skills (commit 981e1d7, MIT). The code is written here.

Two axes define the regime: trend strength (ADX) and volatility (rank of ATR/price against
its own recent history). Grids hold the source's usual values and were fixed before any
result was seen, so that research stays an honest test of them.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradebot import indicators as ind
from tradebot.strategies.base import Strategy, hysteresis

Bools = npt.NDArray[np.bool_]
Floats = npt.NDArray[np.float64]

ATR_PERIOD = 14
EXTREME_VOL_RANK = 0.95


def vol_rank(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Rank of relative ATR within the last ``lookback`` bars: 0 = calmest, 1 = wildest."""
    relative_atr = ind.atr(df, ATR_PERIOD) / df["close"]
    return relative_atr.rolling(lookback, min_periods=lookback).rank(pct=True)


def regime_signal(trending: Bools, hot: Bools, trend: Floats, fade: Floats) -> Floats:
    """Four quadrants: follow a calm trend, half size in a wild one, fade a quiet range,
    stand aside in a wild range."""
    out: Floats = np.select([trending & ~hot, trending & hot, ~trending & ~hot],
                            [trend, trend * 0.5, fade], default=0.0)
    return out


class RegimeSwitch(Strategy):
    name = "regime_switch"

    def __init__(self, adx_min: float = 25.0, vol_pct: float = 0.75, lookback: int = 500,
                 fast: int = 20, slow: int = 100, z_n: int = 48, z_in: float = 2.0) -> None:
        super().__init__(adx_min=adx_min, vol_pct=vol_pct, lookback=lookback, fast=fast, slow=slow,
                         z_n=z_n, z_in=z_in)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        c = df["close"]
        strength = ind.adx(df, ATR_PERIOD)
        rank = vol_rank(df, p["lookback"])
        trend = (ind.ema(c, p["fast"]) - ind.ema(c, p["slow"])).to_numpy(float)
        z = (c - ind.sma(c, p["z_n"])) / c.rolling(p["z_n"], min_periods=p["z_n"]).std()
        fade = hysteresis(z < -p["z_in"], z > 0, z > p["z_in"], z < 0)
        out = regime_signal((strength >= p["adx_min"]).to_numpy(), (rank > p["vol_pct"]).to_numpy(),
                            np.sign(trend), fade.to_numpy(float))
        return pd.Series(out, index=df.index).where(strength.notna() & rank.notna())


def vol_scale(rank: Floats, vol_pct: float) -> Floats:
    """Full size normally, half when volatility is high, flat when it is extreme."""
    out: Floats = np.select([rank > EXTREME_VOL_RANK, rank > vol_pct], [0.0, 0.5], default=1.0)
    return out


class VolScaledTrend(Strategy):
    """EMA-cross direction, sized down as volatility rises."""

    name = "vol_scaled_trend"

    def __init__(self, fast: int = 20, slow: int = 100, vol_pct: float = 0.75, lookback: int = 500) -> None:
        super().__init__(fast=fast, slow=slow, vol_pct=vol_pct, lookback=lookback)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        trend = (ind.ema(df["close"], p["fast"]) - ind.ema(df["close"], p["slow"])).to_numpy(float)
        rank = vol_rank(df, p["lookback"])
        scale = vol_scale(rank.to_numpy(float), p["vol_pct"])
        return pd.Series(np.sign(trend) * scale, index=df.index).where(rank.notna())


def chandelier_positions(direction: Floats, close: Floats, high_max: Floats, low_min: Floats,
                         atr: Floats, mult: float, max_hold: int) -> Floats:
    """Enter when ``direction`` flips; leave on the chandelier level or the time stop.

    A long exits below highest high - mult x ATR, a short above lowest low + mult x ATR. A
    position still not in profit after ``max_hold`` bars also exits. After an exit the
    strategy waits for the next flip, so it does not re-enter the move it just left.
    """
    out = np.zeros(len(close))
    pos, entry, entry_bar = 0.0, 0.0, 0
    for i in range(1, len(close)):
        if pos != 0:
            level = high_max[i] - mult * atr[i] if pos > 0 else low_min[i] + mult * atr[i]
            stopped = (close[i] - level) * pos < 0
            stale = i - entry_bar > max_hold and (close[i] - entry) * pos <= 0
            if stopped or stale:
                pos = 0.0
        flipped = direction[i] != 0 and direction[i] != direction[i - 1]
        if flipped and not np.isnan(atr[i]):
            pos, entry, entry_bar = direction[i], close[i], i
        out[i] = pos
    return out


class ChandelierTrend(Strategy):
    """EMA-cross entries with a trailing chandelier exit and a time stop."""

    name = "chandelier_trend"
    uses_stops = False  # the exit lives in the signal

    def __init__(self, fast: int = 20, slow: int = 100, lookback: int = 22, mult: float = 2.5,
                 max_hold: int = 96) -> None:
        super().__init__(fast=fast, slow=slow, lookback=lookback, mult=mult, max_hold=max_hold)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        n = p["lookback"]
        spread = (ind.ema(df["close"], p["fast"]) - ind.ema(df["close"], p["slow"])).to_numpy(float)
        out = chandelier_positions(
            np.sign(spread), df["close"].to_numpy(float),
            df["high"].rolling(n, min_periods=1).max().to_numpy(float),
            df["low"].rolling(n, min_periods=1).min().to_numpy(float),
            ind.atr(df, ATR_PERIOD).to_numpy(float), p["mult"], p["max_hold"],
        )
        return pd.Series(out, index=df.index)
