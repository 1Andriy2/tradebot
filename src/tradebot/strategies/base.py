"""Strategy interface.

A strategy maps candles to a target exposure in [-1, 1] for every bar, using only data
up to and including that bar's close. +1 = full allowed long, -1 = full allowed short,
0 = flat. The RiskManager turns this into a position size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    name: str = "strategy"
    uses_stops: bool = True  # False: ATR stop/take levels are not applied
    history_bars: int = 0  # candles needed live, when more than the step's usual window

    def __init__(self, **params: Any) -> None:
        self.params = params

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        args = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({args})"

    def fit(self, df: pd.DataFrame) -> None:  # noqa: B027 - optional hook
        """Learn from training data (walk-forward). Most strategies have nothing to fit."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Raw target exposure per bar (may contain NaN during warm-up)."""

    def signal(self, df: pd.DataFrame) -> pd.Series:
        return self.generate(df).reindex(df.index).fillna(0.0).clip(-1.0, 1.0).astype(float)


def hysteresis(
    enter_long: pd.Series,
    exit_long: pd.Series,
    enter_short: pd.Series,
    exit_short: pd.Series,
) -> pd.Series:
    """Stateful position: enter on a trigger and hold until the matching exit trigger."""
    el, xl = enter_long.to_numpy(bool), exit_long.to_numpy(bool)
    es, xs = enter_short.to_numpy(bool), exit_short.to_numpy(bool)
    out = np.zeros(len(el))
    pos = 0.0
    for i in range(len(el)):
        if pos > 0 and xl[i] or pos < 0 and xs[i]:
            pos = 0.0
        if pos == 0.0:
            if el[i]:
                pos = 1.0
            elif es[i]:
                pos = -1.0
        out[i] = pos
    return pd.Series(out, index=enter_long.index)
