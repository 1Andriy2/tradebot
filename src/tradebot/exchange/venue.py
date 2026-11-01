"""Order execution venues. Only demo venues exist; there is no live-trading implementation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import ccxt

from tradebot.exchange.guard import DEMO_HOSTS, install_guard

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    qty: float      # signed, executed
    price: float
    fee: float


class Venue(Protocol):
    def equity(self) -> float: ...
    def position_units(self, symbol: str) -> float: ...
    def prepare(self, symbol: str, leverage: float) -> None: ...
    def market_order(
        self, symbol: str, qty: float, reduce_only: bool, ref_price: float,
        stop_loss: float | None = None, take_profit: float | None = None,
    ) -> Fill | None: ...


class BybitDemoVenue:
    """Bybit Demo Trading via ccxt. Every request is host-checked against the demo allowlist."""

    exchange_id = "bybit"

    def __init__(self, api_key: str, secret: str, fee_rate: float = 0.001) -> None:
        if self.exchange_id not in DEMO_HOSTS:
            raise RuntimeError("no demo allowlist for bybit")
        self.client: Any = ccxt.bybit({
            "apiKey": api_key, "secret": secret, "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        self.client.enable_demo_trading(True)
        install_guard(self.client, self.exchange_id)  # crashes if not a demo endpoint
        self.client.load_markets()
        self.fee_rate = fee_rate

    @classmethod
    def from_env(cls) -> BybitDemoVenue:
        if os.environ.get("EXCHANGE_ID", "bybit") != "bybit":
            raise RuntimeError("EXCHANGE_ID must be 'bybit' (the only approved demo venue)")
        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not secret:
            raise RuntimeError("EXCHANGE_API_KEY / EXCHANGE_API_SECRET are not set")
        return cls(key, secret)

    def equity(self) -> float:
        bal = self.client.fetch_balance()
        try:
            return float(bal["info"]["result"]["list"][0]["totalEquity"])
        except (KeyError, IndexError, TypeError, ValueError):
            return float(bal["total"]["USDT"])

    def position_units(self, symbol: str) -> float:
        units = 0.0
        for p in self.client.fetch_positions([symbol]):
            contracts = float(p.get("contracts") or 0.0) * float(p.get("contractSize") or 1.0)
            units += contracts if p.get("side") == "long" else -contracts
        return units

    def prepare(self, symbol: str, leverage: float) -> None:
        try:
            self.client.set_leverage(leverage, symbol)
        except ccxt.ExchangeError as exc:  # "leverage not modified" is fine
            log.debug("set_leverage: %s", exc)

    def market_order(
        self, symbol: str, qty: float, reduce_only: bool, ref_price: float,
        stop_loss: float | None = None, take_profit: float | None = None,
    ) -> Fill | None:
        market = self.client.market(symbol)
        amount = float(self.client.amount_to_precision(symbol, abs(qty)))
        min_amount = float((market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.0)
        if amount <= 0 or amount < min_amount:
            log.info("%s: order of %.8f below minimum %.8f, skipped", symbol, abs(qty), min_amount)
            return None
        side = "buy" if qty > 0 else "sell"
        params: dict[str, Any] = {"reduceOnly": reduce_only}
        if not reduce_only:
            if stop_loss:
                params["stopLoss"] = {"triggerPrice": self.client.price_to_precision(symbol, stop_loss)}
            if take_profit:
                params["takeProfit"] = {"triggerPrice": self.client.price_to_precision(symbol, take_profit)}
        order = self.client.create_order(symbol, "market", side, amount, None, params)
        price = float(order.get("average") or order.get("price") or ref_price)
        fee = float((order.get("fee") or {}).get("cost") or amount * price * self.fee_rate)
        signed = amount if qty > 0 else -amount
        return Fill(signed, price, fee)
