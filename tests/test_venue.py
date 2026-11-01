from __future__ import annotations

from typing import Any

import pytest

from tradebot.exchange.venue import CcxtDemoVenue


class FakeClient:
    def __init__(self, contract_size: float) -> None:
        self.cs = contract_size
        self.orders: list[tuple[Any, ...]] = []

    def market(self, symbol: str) -> dict[str, Any]:
        return {"contractSize": self.cs, "limits": {"amount": {"min": 0.01}}}

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{int(amount * 100) / 100:.2f}"

    def price_to_precision(self, symbol: str, price: float) -> str:
        return f"{price:.1f}"

    def create_order(self, symbol: str, type_: str, side: str, amount: float, price: Any,
                     params: dict[str, Any]) -> dict[str, Any]:
        self.orders.append((side, amount, params))
        return {"average": 100.0, "fee": {"cost": -0.05}}


def make(exchange_id: str, cs: float) -> CcxtDemoVenue:
    v = CcxtDemoVenue.__new__(CcxtDemoVenue)
    v.exchange_id, v.client, v.fee_rate, v._prepared = exchange_id, FakeClient(cs), 0.001, set()
    return v


def test_okx_order_converts_units_to_contracts() -> None:
    v = make("okx", 0.01)
    fill = v.market_order("BTC/USDT:USDT", -0.5, False, 100.0, stop_loss=110.0, take_profit=90.0)
    side, amount, params = v.client.orders[0]
    assert side == "sell" and amount == pytest.approx(50.0)  # 0.5 BTC = 50 contracts of 0.01
    assert params["marginMode"] == "cross" and params["stopLoss"]["triggerPrice"] == "110.0"
    assert fill is not None and fill.qty == pytest.approx(-0.5) and fill.fee == pytest.approx(0.05)


def test_tiny_order_is_skipped() -> None:
    v = make("bybit", 1.0)
    assert v.market_order("BTC/USDT:USDT", 0.001, False, 100.0) is None
    assert not v.client.orders


def test_unknown_venue_rejected() -> None:
    with pytest.raises(RuntimeError):
        CcxtDemoVenue("kraken", "k", "s")
