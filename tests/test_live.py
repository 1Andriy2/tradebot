from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.agents import Agent
from tradebot.config import Settings
from tradebot.exchange.venue import Fill
from tradebot.live.runner import Trader, seconds_until_next_bar
from tradebot.store import MemoryStore
from tradebot.strategies.base import Strategy

STEP_MS = 900_000
T0 = 1_760_000_000_000 // STEP_MS * STEP_MS


class FakePublic:
    def __init__(self, n: int = 300, price: float = 100.0) -> None:
        self.rows = [[T0 + i * STEP_MS, price, price * 1.001, price * 0.999, price, 1.0] for i in range(n)]
        self.now = T0 + n * STEP_MS + 1_000  # last row is closed

    def add_bar(self, close: float) -> None:
        ts = self.rows[-1][0] + STEP_MS
        prev = self.rows[-1][4]
        self.rows.append([ts, prev, max(prev, close) * 1.001, min(prev, close) * 0.999, close, 1.0])
        self.now = ts + STEP_MS + 1_000

    def parse_timeframe(self, tf: str) -> int:
        return 900

    def milliseconds(self) -> int:
        return self.now

    def fetch_ohlcv(self, symbol: str, tf: str, since: int, limit: int) -> list[list[float]]:
        return [r for r in self.rows if r[0] >= since][:limit]


class FakeVenue:
    def __init__(self, equity: float = 10_000.0) -> None:
        self._equity = equity
        self.units: dict[str, float] = {}
        self.orders: list[dict[str, object]] = []

    def equity(self) -> float:
        return self._equity

    def position_units(self, symbol: str) -> float:
        return self.units.get(symbol, 0.0)

    def prepare(self, symbol: str, leverage: float) -> None:
        pass

    def market_order(self, symbol: str, qty: float, reduce_only: bool, ref_price: float,
                     stop_loss: float | None = None, take_profit: float | None = None) -> Fill:
        self.orders.append({"symbol": symbol, "qty": qty, "reduce_only": reduce_only,
                            "sl": stop_loss, "tp": take_profit})
        self.units[symbol] = self.units.get(symbol, 0.0) + qty
        return Fill(qty, ref_price, abs(qty) * ref_price * 0.001)


class Const(Strategy):
    name = "const"

    def __init__(self, value: float) -> None:
        super().__init__(value=value)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.params["value"], index=df.index)


def make_trader(venue: FakeVenue | None, public: FakePublic, value: float = 1.0
                ) -> tuple[Trader, MemoryStore]:
    settings = Settings()
    settings.markets.symbols = ["BTC/USDT:USDT"]
    store = MemoryStore()
    agents = [Agent("ma_cross", "baseline", Const(value)), Agent("flat", "baseline", Const(0.0))]
    return Trader(settings, store, public, venue, agents=agents), store


def test_shadow_step_is_idempotent() -> None:
    public = FakePublic()
    trader, store = make_trader(None, public)
    first = trader.step()
    second = trader.step()
    assert first.decisions == 2 and second.decisions == 0
    assert not first.errors
    assert store.select("bot_status")[0]["mode"] == "shadow-only"


def test_shadow_paper_book_fills_on_next_bar() -> None:
    public = FakePublic()
    trader, store = make_trader(None, public)
    trader.step()
    public.add_bar(101.0)
    trader.step()
    pos = store.select("positions", {"agent_id": "ma_cross", "venue": "shadow"})[0]
    assert pos["units"] > 0


def test_champion_trades_demo_with_stops_once_per_bar() -> None:
    public, venue = FakePublic(), FakeVenue()
    trader, store = make_trader(venue, public)
    trader.step()
    trader.step()  # same bar: no second order
    assert len(venue.orders) == 1
    order = venue.orders[0]
    assert order["qty"] > 0 and not order["reduce_only"]
    assert order["sl"] is not None and order["tp"] is not None
    # Notional = 10% * 1x of equity (one symbol): 1000 USDT at price 100.
    assert venue.units["BTC/USDT:USDT"] * 100 == pytest.approx(1000.0)
    assert any(d["venue"] == "demo" and d["action"] == "open" for d in store.select("decisions"))


def test_exchange_stop_is_recorded_and_not_reentered() -> None:
    public, venue = FakePublic(), FakeVenue()
    trader, store = make_trader(venue, public)
    trader.step()
    venue.units["BTC/USDT:USDT"] = 0.0  # attached stop-loss fired on the exchange
    public.add_bar(90.0)
    trader.step()
    trades = store.select("trades", {"venue": "demo"})
    assert len(trades) == 1 and trades[0]["exit_reason"] == "stop_loss"
    assert len(venue.orders) == 1  # unchanged signal does not re-enter after a stop


def test_kill_switch_closes_demo_position() -> None:
    public, venue = FakePublic(), FakeVenue()
    trader, store = make_trader(venue, public)
    trader.step()
    venue._equity = 7_000.0  # 30% drawdown
    public.add_bar(100.0)
    trader.step()
    assert venue.orders[-1]["reduce_only"] is True
    assert venue.units["BTC/USDT:USDT"] == pytest.approx(0.0)
    assert any(e["type"] == "kill_switch" for e in store.select("events"))


def test_flip_closes_then_opens() -> None:
    public, venue = FakePublic(), FakeVenue()
    trader, _ = make_trader(venue, public)
    trader.step()
    trader.agents[0] = Agent("ma_cross", "baseline", Const(-1.0))
    public.add_bar(100.0)
    trader.step()
    assert [o["reduce_only"] for o in venue.orders] == [False, True, False]
    assert venue.units["BTC/USDT:USDT"] < 0


def test_seconds_until_next_bar() -> None:
    assert seconds_until_next_bar(900 * 10 + 100, 900, 15) == pytest.approx(815)
    assert np.isfinite(seconds_until_next_bar(0, 900))


def test_demo_failure_does_not_stop_other_agents() -> None:
    public, venue = FakePublic(), FakeVenue()

    def broken(*args: object, **kwargs: object) -> Fill:
        raise RuntimeError("This API Key does not have trading permission")

    venue.market_order = broken  # type: ignore[method-assign]
    settings = Settings()
    settings.markets.symbols = ["BTC/USDT:USDT"]
    store = MemoryStore()
    agents = [Agent("ma_cross", "baseline", Const(1.0)), Agent("other", "baseline", Const(-1.0))]
    report = Trader(settings, store, public, venue, agents=agents).step()
    assert len(report.errors) == 1 and "ma_cross" in report.errors[0]
    assert store.select("agent_state", {"agent_id": "other", "venue": "shadow"})
    assert store.select("bot_status")[0]["last_error"]



def test_manual_kill_switch_reset_reenables_trading() -> None:
    from tradebot.cli import build_parser, cmd_reset_kill_switch

    public, venue = FakePublic(), FakeVenue()
    trader, store = make_trader(venue, public)
    trader.step()
    venue._equity = 7_000.0  # trips the kill switch
    public.add_bar(100.0)
    trader.step()
    assert venue.units["BTC/USDT:USDT"] == pytest.approx(0.0)

    args = build_parser().parse_args(["reset-kill-switch", "--agent", "ma_cross", "--venue", "demo",
                                      "--symbol", "BTC/USDT:USDT", "--equity", "7000"])
    import tradebot.cli as cli
    cli_store = cli._store
    cli._store = lambda _args: store  # type: ignore[assignment]
    try:
        assert cmd_reset_kill_switch(Settings(), args) == 0
    finally:
        cli._store = cli_store  # type: ignore[assignment]
    public.add_bar(100.0)
    trader.step()
    assert venue.units["BTC/USDT:USDT"] > 0
