from __future__ import annotations

from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import pytest

from tradebot.data import market_stats as ms
from tradebot.data.sync import _paths
from tradebot.research.events import MIN_EVENTS, event_study, extreme, liquidation_studies

HOUR = 3_600_000
T0 = 1_780_000_000_000 // HOUR * HOUR
SYMBOL = "BTC/USDT:USDT"
Rows = list[dict[str, Any]]


class FakeClient:
    def __init__(self, hours: int = 250) -> None:
        self.hours = hours
        self.calls = 0

    def milliseconds(self) -> int:
        return T0 + self.hours * HOUR

    def _rows(self, since: int, limit: int, make: Any) -> list[dict[str, Any]]:
        self.calls += 1
        stamps = [T0 + i * HOUR for i in range(self.hours)]
        return [make(t) for t in stamps if t >= since][:limit]

    def load_markets(self) -> None:
        pass

    def fetch_open_interest_history(self, symbol: str, tf: str, since: int, limit: int) -> Rows:
        return self._rows(since, limit, lambda t: {"timestamp": t, "openInterestValue": 1e9})

    def fetch_long_short_ratio_history(self, symbol: str, tf: str, since: int, limit: int) -> Rows:
        return self._rows(since, limit, lambda t: {"timestamp": t, "longShortRatio": 1.4})

    def market(self, symbol: str) -> dict[str, Any]:
        return {"base": "BTC", "quote": "USDT", "contractSize": 0.01}


def order(ts: int, side: str = "long", price: float = 80_000.0, size: float = 2.0) -> dict[str, str]:
    return {"ts": str(ts), "posSide": side, "bkPx": str(price), "sz": str(size)}


class FakeOrders:
    """Serves pages of orders older than the ``after`` cursor, newest first."""

    def __init__(self, orders: list[dict[str, str]]) -> None:
        self.orders = sorted(orders, key=lambda o: -int(o["ts"]))
        self.pages = 0

    def __call__(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        self.pages += 1
        after = int(params.get("after", 2**62))
        rows = [o for o in self.orders if int(o["ts"]) < after][: ms.PAGE]
        return {"code": "0", "data": [{"details": rows}]}


def test_stats_cache_grows_without_duplicates(tmp_path: Path) -> None:
    client = FakeClient(250)
    first = ms.update_stats(client, tmp_path, SYMBOL)
    assert len(first) == 250 and first.notna().all().all()
    client.hours = 260
    second = ms.update_stats(client, tmp_path, SYMBOL)
    assert len(second) == 260 and second.index.is_unique and second.index.is_monotonic_increasing


def test_stats_paging_stops_when_nothing_is_new(tmp_path: Path) -> None:
    client = FakeClient(5)
    ms.update_stats(client, tmp_path, SYMBOL)
    assert client.calls <= 4


def test_stats_fill_a_gap_on_the_next_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "RETRY_DELAY", 0)
    client = FakeClient(150)
    healthy = client.fetch_long_short_ratio_history

    def down(symbol: str, tf: str, since: int, limit: int) -> Rows:
        raise ccxt.RequestTimeout("rubik timeout")

    client.fetch_long_short_ratio_history = down  # type: ignore[method-assign]
    first = ms.update_stats(client, tmp_path, SYMBOL)
    assert first["open_interest"].notna().all() and first["long_short_ratio"].isna().all()
    client.fetch_long_short_ratio_history = healthy  # type: ignore[method-assign]
    second = ms.update_stats(client, tmp_path, SYMBOL)
    assert len(second) == 150 and second.notna().all().all()


def test_liquidations_values_and_incremental_update(tmp_path: Path) -> None:
    client = FakeClient()
    orders = [order(T0 + i * 1000) for i in range(250)]
    first = ms.update_liquidations(client, tmp_path, SYMBOL, FakeOrders(orders))
    assert len(first) == 250
    assert first["usd"].iloc[0] == pytest.approx(2.0 * 0.01 * 80_000)
    orders.append(order(T0 + 999_000, "short"))
    feed = FakeOrders(orders)
    second = ms.update_liquidations(client, tmp_path, SYMBOL, feed)
    assert len(second) == 251 and feed.pages == 1


def test_liquidation_paging_survives_a_stuck_cursor() -> None:
    page = {"code": "0", "data": [{"details": [order(T0), order(T0 + 1)]}]}
    calls = []

    def stuck(url: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append(params.get("after"))
        return page

    assert len(ms.fetch_liquidations(stuck, "BTC-USDT", 0)) == 2
    assert len(calls) == 2


def test_empty_liquidation_feed_is_fine(tmp_path: Path) -> None:
    out = ms.update_liquidations(FakeClient(), tmp_path, SYMBOL, FakeOrders([]))
    assert out.empty and ms.hourly_liquidations(out).empty


def test_http_get_raises_on_exchange_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {"code": "51000", "msg": "bad"}

    monkeypatch.setattr(ms.httpx, "get", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="51000"):
        ms.http_get("https://example.invalid", {})


def test_hourly_liquidations_split_by_side() -> None:
    frame = pd.DataFrame({"ts": [T0, T0 + 1, T0 + HOUR], "side": ["long", "short", "long"],
                          "usd": [10.0, 3.0, 5.0]})
    table = ms.hourly_liquidations(frame)
    assert table["long"].tolist() == [10.0, 5.0] and table["short"].tolist() == [3.0, 0.0]


def test_new_caches_are_synced(tmp_path: Path) -> None:
    names = {p.name for p in _paths(tmp_path, [SYMBOL], [])}
    assert {ms.stats_path(tmp_path, SYMBOL).name, ms.liquidations_path(tmp_path, SYMBOL).name} <= names


def hours(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")


def walk(n: int, seed: int) -> pd.Series:
    r = np.random.default_rng(seed).normal(0, 0.005, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=hours(n))


def test_event_study_reports_too_few_events() -> None:
    close = walk(2000, 1)
    events = pd.Series(False, index=close.index)
    events.iloc[: (MIN_EVENTS - 1) * 30: 30] = True
    assert event_study(close, events, 4)["status"] == "not_enough_data"


def test_event_study_counts_overlapping_events_once() -> None:
    close = walk(2000, 2)
    events = pd.Series(False, index=close.index)
    events.iloc[100:110] = True
    assert event_study(close, events, 24)["n"] == 1


def test_event_study_false_positive_rate() -> None:
    hits = 0
    for seed in range(60):
        close = walk(3000, seed)
        events = pd.Series(np.random.default_rng(seed + 500).random(3000) < 0.02, index=close.index)
        hits += event_study(close, events, 4, seed)["p_value"] < 0.05
    assert hits <= 9


def test_event_study_finds_a_real_reversal() -> None:
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.003, 4000)
    positions = np.arange(50, 3900, 50)
    for t in positions:
        r[t + 1: t + 5] += 0.006
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=hours(4000))
    events = pd.Series(False, index=close.index)
    events.iloc[positions] = True
    out = event_study(close, events, 4)
    assert out["status"] == "tested" and out["p_value"] < 0.01 and out["mean_after"] > out["base_mean"]


def test_extreme_uses_only_earlier_hours() -> None:
    s = pd.Series(np.random.default_rng(3).random(1000), index=hours(1000))
    before = extreme(s).iloc[:700]
    changed = s.copy()
    changed.iloc[700:] = 1e9
    assert (extreme(changed).iloc[:700] == before).all()


def test_liquidation_studies_handle_missing_data() -> None:
    close = walk(500, 4)
    rows = liquidation_studies(close, ms.hourly_liquidations(pd.DataFrame()), pd.DataFrame())
    assert rows and all(r["status"] == "not_enough_data" for r in rows)
