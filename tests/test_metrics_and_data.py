from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot import indicators as ind
from tradebot.config import load_settings
from tradebot.data.history import fetch_ohlcv_range, load_cached, update_cache
from tradebot.metrics import compute_metrics, max_drawdown

HOUR_MS = 3_600_000


def test_max_drawdown() -> None:
    eq = pd.Series([100, 120, 90, 130, 65], dtype=float)
    assert max_drawdown(eq) == pytest.approx(0.5)


def test_compute_metrics_basic() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    eq = pd.Series([100, 101, 102, 101, 103], index=idx, dtype=float)
    trades = pd.DataFrame({"pnl": [2.0, -1.0, 2.0]})
    close = pd.Series([10, 10, 10, 10, 11], index=idx, dtype=float)
    m = compute_metrics(eq, trades, close)
    assert m["total_return"] == pytest.approx(0.03)
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(4.0)
    assert m["num_trades"] == 3
    assert m["excess_vs_buy_hold"] == pytest.approx(0.03 - 0.1)
    assert m["sharpe"] > 0 and m["sortino"] > 0


def test_rsi_bounds() -> None:
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 500).cumsum() + 100)
    r = ind.rsi(s).dropna()
    assert ((r >= 0) & (r <= 100)).all()


class FakeClient:
    """Mimics ccxt pagination over an hourly series of 50 candles."""

    def __init__(self, n: int = 50, start_ms: int = 1_700_000_000_000 // HOUR_MS * HOUR_MS) -> None:
        self.start = start_ms
        self.rows = [[start_ms + i * HOUR_MS, 1, 2, 0.5, 1.5, 10] for i in range(n)]
        self.now = start_ms + (n - 1) * HOUR_MS + 1  # the last candle is still open

    def parse_timeframe(self, tf: str) -> int:
        return 3600

    def milliseconds(self) -> int:
        return self.now

    def fetch_ohlcv(self, symbol: str, tf: str, since: int, limit: int) -> list[list[float]]:
        return [r for r in self.rows if r[0] >= since][:7]


def test_fetch_paginates_and_drops_open_candle() -> None:
    c = FakeClient()
    df = fetch_ohlcv_range(c, "BTC/USDT:USDT", "1h", c.start)
    assert len(df) == 49
    assert df.index.is_monotonic_increasing and df.index.is_unique


def test_fetch_retries_transient_errors() -> None:
    import ccxt

    c = FakeClient(n=10)
    original = c.fetch_ohlcv
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> list[list[float]]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ccxt.ExchangeError("svc error: Get kline failed")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    c.fetch_ohlcv = flaky  # type: ignore[method-assign]
    df = fetch_ohlcv_range(c, "BTC/USDT:USDT", "1h", c.start, sleep=lambda _: None)
    assert len(df) == 9


def test_update_cache_is_incremental(tmp_path: Path) -> None:
    c = FakeClient(n=20)
    start = pd.Timestamp(c.start, unit="ms", tz="UTC").isoformat()
    first = update_cache(c, tmp_path, "BTC/USDT:USDT", "1h", start)
    c.rows += [[c.start + i * HOUR_MS, 1, 2, 0.5, 1.5, 10] for i in range(20, 30)]
    c.now = c.start + 29 * HOUR_MS + 1
    second = update_cache(c, tmp_path, "BTC/USDT:USDT", "1h", start)
    assert len(first) == 19 and len(second) == 29
    pd.testing.assert_frame_equal(load_cached(tmp_path, "BTC/USDT:USDT", "1h"), second,
                                  check_freq=False)


def test_settings_load_defaults() -> None:
    s = load_settings()
    assert s.venue.exchange_id == "bybit"
    assert s.costs.fee_rate == pytest.approx(0.001)
