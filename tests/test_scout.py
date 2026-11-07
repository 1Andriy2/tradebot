from __future__ import annotations

import json
import zlib
from typing import Any

import numpy as np
import pandas as pd

from tradebot.config import Settings
from tradebot.data import universe
from tradebot.live.runner import Trader
from tradebot.live.scout import PREFIX, ScoutTrader, run_scout, scout_agents
from tradebot.store import MemoryStore

NOW = pd.Timestamp("2026-11-07 12:00", tz="UTC")
BAR_MS = 300_000


class Exchange:
    """Public client with one price series per symbol, closing a new bar on demand."""

    def __init__(self, bars: int = 400) -> None:
        self.bars = bars
        self.now_ms = int(pd.Timestamp("2026-11-07", tz="UTC").timestamp() * 1000)

    def parse_timeframe(self, _: str) -> int:
        return 300

    def milliseconds(self) -> int:
        return self.now_ms

    def advance(self, bars: int = 1) -> None:
        self.now_ms += bars * BAR_MS

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 5000)))
        start = self.now_ms - self.bars * BAR_MS
        rows = []
        for i, close in enumerate(closes):
            ts = start + i * BAR_MS
            if since <= ts <= self.now_ms:
                rows.append([ts, close, close * 1.002, close * 0.998, close, 1.0])
        return rows[:limit]


def trader(store: MemoryStore, client: Exchange, coins: list[str]) -> ScoutTrader:
    return ScoutTrader(Settings(), store, client, coins)


def test_lane_is_paper_only_and_cannot_touch_the_main_books() -> None:
    bot = trader(MemoryStore(), Exchange(), ["SOL/USDT:USDT"])
    assert bot.venue is None
    assert all(a.id.startswith(PREFIX) and a.kind == "baseline" for a in scout_agents())
    assert "s5_chandelier_trend_4h" not in {a.id for a in scout_agents()}


def test_a_step_stores_no_holds_or_candles() -> None:
    store, client = MemoryStore(), Exchange()
    bot = trader(store, client, ["SOL/USDT:USDT"])
    bot.step()
    client.advance(3)
    report = bot.step()
    assert report.errors == []
    assert report.decisions > 0
    assert all(d["action"] != "hold" for d in store.select("decisions"))
    assert store.select("candles") == []


def test_repeating_a_bar_does_nothing() -> None:
    store, client = MemoryStore(), Exchange()
    bot = trader(store, client, ["SOL/USDT:USDT"])
    bot.step()
    client.advance()
    assert bot.step().decisions > 0
    assert bot.step().decisions == 0


def test_a_restart_replays_from_the_saved_bar() -> None:
    store, client = MemoryStore(), Exchange()
    trader(store, client, ["SOL/USDT:USDT"]).step()
    saved = {r["agent_id"]: r["last_bar"] for r in store.select("agent_state")}
    assert saved
    client.advance(2)
    restarted = trader(store, client, ["SOL/USDT:USDT"])
    assert restarted.step().decisions >= 2 * len(saved)


def test_a_coin_that_leaves_the_list_is_closed_and_then_dropped() -> None:
    store, client = MemoryStore(), Exchange()
    bot = trader(store, client, ["SOL/USDT:USDT"])
    bot.step()  # buy_hold is long from its first bar
    client.advance()
    bot.step()
    held = bot.rows[("s5_buy_hold", "SOL/USDT:USDT")]["state"]["book"]["units"]
    assert held > 0
    bot.coins = ["XRP/USDT:USDT"]
    client.advance(2)
    bot.step()
    assert "SOL/USDT:USDT" in bot.symbols
    assert bot.rows[("s5_buy_hold", "SOL/USDT:USDT")]["state"]["book"]["units"] == 0
    client.advance()
    bot.step()
    assert "SOL/USDT:USDT" not in bot.symbols


def test_equity_counts_a_dropped_coin_and_is_written_once_an_hour() -> None:
    store, client = MemoryStore(), Exchange()
    bot = trader(store, client, ["SOL/USDT:USDT", "XRP/USDT:USDT"])
    bot.step()
    first = len(store.select("equity_snapshots"))
    client.advance()
    bot.step()
    assert first == len(bot.agents)
    assert len(store.select("equity_snapshots")) == first


def test_the_main_step_still_stores_holds() -> None:
    store = MemoryStore()
    plain = Trader.__new__(Trader)
    plain.store = store
    plain._record([{"action": "hold"}])
    assert store.select("decisions") == [{"action": "hold"}]


class Clock:
    def __init__(self, start: float) -> None:
        self.now = start
        self.naps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.now += seconds


class CountingTrader:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls: list[float] = []
        self.fail_first = fail_first

    def step(self) -> None:
        self.calls.append(0.0)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("network")


def test_the_loop_looks_each_minute_and_steps_once_a_bar() -> None:
    clock, bot = Clock(1_800_000_000.0), CountingTrader()
    run_scout(bot, 30, clock, clock.sleep)  # type: ignore[arg-type]
    assert len(bot.calls) == 6
    assert all(n <= 65 for n in clock.naps)
    assert len(clock.naps) >= 29


def test_a_failed_step_is_tried_again_and_does_not_end_the_loop() -> None:
    clock, bot = Clock(1_800_000_000.0), CountingTrader(fail_first=True)
    run_scout(bot, 4, clock, clock.sleep)  # type: ignore[arg-type]
    assert len(bot.calls) == 2


class Markets:
    def __init__(self, coins: list[str]) -> None:
        self.coins = coins

    def load_markets(self) -> dict[str, dict[str, Any]]:
        rows = {f"{c}/USDT:USDT": {"base": c, "symbol": f"{c}/USDT:USDT", "swap": True, "linear": True,
                                   "quote": "USDT", "active": True} for c in self.coins}
        rows["OLD/USDT:USDT"] = {**rows[f"{self.coins[0]}/USDT:USDT"], "base": "OLD", "active": False}
        return rows

    def fetch_tickers(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        return {s: {"quoteVolume": float(i)} for i, s in enumerate(symbols)}


COINS = [f"C{i}" for i in range(12)]


def counts_answer(_: str) -> list[dict[str, Any]]:
    rows = [{"symbol": f"{c}USDT", "count": 100 + i} for i, c in enumerate(COINS)]
    return [*rows, {"symbol": "OLDUSDT", "count": 10**9}, {"symbol": "C0BTC", "count": 10**9},
            {"symbol": "1000C1USDT", "count": 5000}, {"symbol": "NOTHEREUSDT", "count": 10**9}]


def test_coins_are_ranked_by_trades_and_must_be_live_on_the_venue() -> None:
    picked = universe.pick(Markets(COINS), NOW, counts_answer)
    symbols = [c["symbol"] for c in picked["coins"]]
    assert picked["source"] == "binance_futures"
    assert len(symbols) == universe.COINS
    assert symbols[0] == "C1/USDT:USDT"  # 1000C1 is C1, with the larger count
    assert symbols[1] == "C11/USDT:USDT"
    assert "OLD/USDT:USDT" not in symbols


def test_spot_counts_stand_in_and_volume_is_the_last_resort() -> None:
    def spot_only(url: str) -> list[dict[str, Any]]:
        if "fapi" in url:
            raise RuntimeError("451")
        return counts_answer(url)

    def nothing(_: str) -> list[dict[str, Any]]:
        raise RuntimeError("down")

    assert universe.pick(Markets(COINS), NOW, spot_only)["source"] == "binance_spot"
    assert universe.pick(Markets(COINS), NOW, nothing)["source"] == "venue_volume"


def test_the_list_is_picked_again_only_after_a_week() -> None:
    store, client = MemoryStore(), Markets(COINS)
    first = universe.current(store, client, NOW, counts_answer)
    client.coins = COINS[:11]
    assert universe.current(store, client, NOW + pd.Timedelta("6D"), counts_answer) == first
    later = universe.current(store, client, NOW + pd.Timedelta("7D"), counts_answer)
    assert "C11/USDT:USDT" not in later
    assert len(store.select("events")) == 2


def test_a_failed_pick_keeps_the_old_list_or_the_fallback() -> None:
    class Broken:
        def load_markets(self) -> dict[str, Any]:
            raise RuntimeError("down")

    store = MemoryStore()
    assert universe.current(store, Broken(), NOW, counts_answer) == universe.FALLBACK
    good = universe.current(store, Markets(COINS), NOW, counts_answer)
    saved = json.loads(store.download(universe.BUCKET, universe.FILE))
    assert [c["symbol"] for c in saved["coins"]] == good
    assert universe.current(store, Broken(), NOW + pd.Timedelta("8D"), counts_answer) == good


def test_coins_pegged_to_the_dollar_are_left_out() -> None:
    assert "USDC" not in universe.trade_counts([{"symbol": "USDCUSDT", "count": 10**9}])
    assert universe.trade_counts([{"symbol": "SOLUSDT", "count": 5}]) == {"SOL": 5}


def test_a_list_picked_under_older_rules_is_picked_again() -> None:
    store = MemoryStore()
    coin = {"symbol": "USDC/USDT:USDT", "activity": 1}
    old = {"picked_at": NOW.isoformat(), "source": "x", "coins": [coin]}
    store.upload(universe.BUCKET, universe.FILE, json.dumps(old).encode())
    assert "USDC/USDT:USDT" not in universe.current(store, Markets(COINS), NOW, counts_answer)
