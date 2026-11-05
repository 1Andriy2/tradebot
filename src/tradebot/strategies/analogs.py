"""Historical analogs: what happened after moments that looked like now.

Each bar is described by a few normalised features. The nearest moments in the training data
give a distribution of forward returns, and the strategy trades only when it leans clearly
one way and the expected move covers costs.

A past moment counts only if its outcome was already known at the bar being judged. Without
this rule a bar inside the training window would find itself and read its own future.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradebot import indicators as ind
from tradebot.strategies.base import Strategy

Floats = npt.NDArray[np.float64]

MAX_TRAIN_POINTS = 6000  # keeps the search fast; a thinned sample still covers every regime
MIN_EXPECTED_MOVE = 0.003  # about one round trip of fees and slippage
QUERY_FACTOR = 3  # extra neighbours fetched, since some are dropped by the known-outcome rule
CHUNK = 20_000


def state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Where the market is now, using data up to each bar's close only."""
    c = df["close"]
    atr = ind.atr(df, 14)
    relative_atr = atr / c
    return pd.DataFrame({
        "momentum_short": (c / c.shift(4) - 1) / relative_atr,
        "momentum_long": (c / c.shift(24) - 1) / relative_atr,
        "trend": (c - ind.ema(c, 96)) / atr,
        "volatility": np.log(relative_atr / relative_atr.rolling(500, min_periods=100).median()),
        "strength": ind.adx(df, 14) / 100,
        "rsi": ind.rsi(c, 14) / 100,
    }, index=df.index).replace([np.inf, -np.inf], np.nan)


class AnalogIndex:
    """Searchable past states with their forward outcomes."""

    def __init__(self, features: pd.DataFrame, outcomes: pd.DataFrame, known_at: pd.Series) -> None:
        from scipy.spatial import cKDTree

        step = max(1, len(features) // MAX_TRAIN_POINTS)
        rows = features.iloc[::step]
        self.mean, self.std = features.mean(), features.std().replace(0, 1.0)
        self.tree = cKDTree(((rows - self.mean) / self.std).to_numpy())
        self.outcomes = outcomes.iloc[::step].to_numpy(float)
        self.known_at = known_at.iloc[::step].to_numpy("datetime64[ns]")

    def neighbours(self, states: pd.DataFrame, k: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Up to ``k`` nearest usable rows per state, and a mask of which slots are usable."""
        n_query = min(k * QUERY_FACTOR, len(self.outcomes))
        scaled = ((states - self.mean) / self.std).to_numpy()
        _, idx = self.tree.query(scaled, k=n_query)
        idx = idx.reshape(len(states), n_query)
        times = states.index.to_numpy("datetime64[ns]")[:, None]
        usable = self.known_at[idx] <= times
        return idx, usable & (usable.cumsum(axis=1) <= k)


def build_index(df: pd.DataFrame, horizon: int) -> AnalogIndex | None:
    """Index of ``df`` for forward returns over ``horizon`` bars."""
    features = state_features(df)
    forward = (df["close"].shift(-horizon) / df["close"] - 1).rename("forward")
    known_at = pd.Series(df.index, index=df.index).shift(-horizon)
    keep = features.notna().all(axis=1) & forward.notna()
    if keep.sum() < 50:
        return None
    return AnalogIndex(features[keep], forward[keep].to_frame(), known_at[keep])


class AnalogForecast(Strategy):
    name = "analog"
    uses_stops = False  # the forecast decides when to leave

    def __init__(self, k: int = 100, horizon: int = 16, edge: float = 0.1) -> None:
        super().__init__(k=k, horizon=horizon, edge=edge)
        self.index: AnalogIndex | None = None

    def fit(self, df: pd.DataFrame) -> None:
        self.index = build_index(df, self.params["horizon"])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(0.0, index=df.index)
        if self.index is None:
            return out
        k, edge = self.params["k"], self.params["edge"]
        states = state_features(df).dropna()
        for start in range(0, len(states), CHUNK):
            chunk = states.iloc[start:start + CHUNK]
            idx, chosen = self.index.neighbours(chunk, k)
            forward = self.index.outcomes[idx, 0]
            count = chosen.sum(axis=1)
            enough = count >= k // 2
            safe = np.maximum(count, 1)
            p_up = ((forward > 0) & chosen).sum(axis=1) / safe
            mean = (forward * chosen).sum(axis=1) / safe
            long = enough & (p_up >= 0.5 + edge) & (mean >= MIN_EXPECTED_MOVE)
            short = enough & (p_up <= 0.5 - edge) & (mean <= -MIN_EXPECTED_MOVE)
            out.loc[chunk.index] = long.astype(float) - short.astype(float)
        return out


def scenario_outlook(df: pd.DataFrame, horizon: int, k: int = 200) -> dict[str, Any] | None:
    """Spread of outcomes after the moments most like the last bar.

    This describes the past. Its worth as a forecast is what the ``analog`` family's research
    results measure, so read the two together.
    """
    c, low, high = df["close"], df["low"], df["high"]
    lows = low[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    highs = high[::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    outcomes = pd.DataFrame({"forward": c.shift(-horizon) / c - 1, "worst": lows / c - 1,
                             "best": highs / c - 1})
    features = state_features(df)
    known_at = pd.Series(df.index, index=df.index).shift(-horizon)
    keep = features.notna().all(axis=1) & outcomes.notna().all(axis=1)
    now = features.dropna().tail(1)
    if keep.sum() < k or now.empty:
        return None
    index = AnalogIndex(features[keep], outcomes[keep], known_at[keep])
    idx, chosen = index.neighbours(now, k)
    rows = index.outcomes[idx[0][chosen[0]]]
    forward, worst, best = rows[:, 0], rows[:, 1], rows[:, 2]
    base = outcomes.loc[keep, "forward"]
    return {
        "as_of": now.index[-1].isoformat(), "horizon_bars": horizon, "n": len(rows),
        "p_up": float(np.mean(forward > 0)), "base_p_up": float((base > 0).mean()),
        "q10": float(np.quantile(forward, 0.1)), "q50": float(np.quantile(forward, 0.5)),
        "q90": float(np.quantile(forward, 0.9)),
        "typical_worst": float(np.median(worst)), "typical_best": float(np.median(best)),
    }
