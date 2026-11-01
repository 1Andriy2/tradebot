from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(close: list[float] | np.ndarray, freq: str = "1h", spread: float = 0.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq=freq, tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx
    )


@pytest.fixture
def random_walk() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000)))
    return make_ohlcv(close, spread=0.002)
