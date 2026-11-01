"""Order execution venues. Only demo venues exist; there is no live-trading implementation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import ccxt

from tradebot.exchange.guard import DEMO_ENDPOINTS, install_guard

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fill:
    qty: float      # signed, executed, in base-asset units
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


class CcxtDemoVenue:
    """USDT perpetuals on a ccxt demo environment (Bybit Demo Trading or OKX Demo Trading).

    Every request is checked against the demo allowlist in ``guard``; construction fails if
    the client is not in demo mode.
    """

    def __init__(self, exchange_id: str, api_key: str, secret: str, password: str | None = None,
                 fee_rate: float = 0.001) -> None:
        if exchange_id not in DEMO_ENDPOINTS:
            raise RuntimeError(f"no demo allowlist for '{exchange_id}'")
        self.exchange_id = exchange_id
        config: dict[str, Any] = {"apiKey": api_key, "secret": secret, "enableRateLimit": True,
                                  "options": {"defaultType": "swap"}}
        if password:
            config["password"] = password
        self.client: Any = getattr(ccxt, exchange_id)(config)
        if exchange_id == "bybit":
            self.client.enable_demo_trading(True)
        elif exchange_id == "okx":
            self.client.set_sandbox_mode(True)  # adds x-simulated-trading: 1
        install_guard(self.client, exchange_id)  # crashes if not a demo endpoint
        self.client.load_markets()
        self.fee_rate = fee_rate
        self._prepared: set[str] = set()

    @classmethod
    def from_env(cls) -> CcxtDemoVenue:
        exchange_id = os.environ.get("EXCHANGE_ID", "bybit")
        key, secret = os.environ.get("EXCHANGE_API_KEY"), os.environ.get("EXCHANGE_API_SECRET")
        if not key or not secret:
            raise RuntimeError("EXCHANGE_API_KEY / EXCHANGE_API_SECRET are not set")
        password = os.environ.get("EXCHANGE_API_PASSPHRASE") or None
        if exchange_id == "okx" and not password:
            raise RuntimeError("EXCHANGE_API_PASSPHRASE is required for OKX")
        return cls(exchange_id, key, secret, password)

    def equity(self) -> float:
        info = self.client.fetch_balance()
        raw = info.get("info", {})
        try:
            if self.exchange_id == "bybit":
                return float(raw["result"]["list"][0]["totalEquity"])
            if self.exchange_id == "okx":
                return float(raw["data"][0]["totalEq"])
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        return float(info["total"]["USDT"])

    def _contract_size(self, symbol: str) -> float:
        return float(self.client.market(symbol).get("contractSize") or 1.0)

    def position_units(self, symbol: str) -> float:
        units = 0.0
        for p in self.client.fetch_positions([symbol]):
            size = float(p.get("contracts") or 0.0) * float(p.get("contractSize") or 1.0)
            units += size if p.get("side") == "long" else -size
        return units

    def prepare(self, symbol: str, leverage: float) -> None:
        params = {"mgnMode": "cross"} if self.exchange_id == "okx" else {}
        if self.exchange_id == "okx" and "_mode" not in self._prepared:
            try:
                self.client.set_position_mode(False)  # one-way (net) positions
            except ccxt.ExchangeError as exc:
                log.debug("set_position_mode: %s", exc)
            self._prepared.add("_mode")
        try:
            self.client.set_leverage(leverage, symbol, params)
        except ccxt.ExchangeError as exc:  # "leverage not modified" is fine
            log.debug("set_leverage: %s", exc)

    def market_order(
        self, symbol: str, qty: float, reduce_only: bool, ref_price: float,
        stop_loss: float | None = None, take_profit: float | None = None,
    ) -> Fill | None:
        market = self.client.market(symbol)
        cs = self._contract_size(symbol)
        contracts = float(self.client.amount_to_precision(symbol, abs(qty) / cs))
        min_amount = float((market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.0)
        if contracts <= 0 or contracts < min_amount:
            log.info("%s: order of %.8f contracts below minimum %.8f, skipped",
                     symbol, abs(qty) / cs, min_amount)
            return None
        side = "buy" if qty > 0 else "sell"
        params: dict[str, Any] = {"reduceOnly": reduce_only}
        if self.exchange_id == "okx":
            params["marginMode"] = "cross"
        if not reduce_only:
            if stop_loss:
                params["stopLoss"] = {"triggerPrice": self.client.price_to_precision(symbol, stop_loss)}
            if take_profit:
                params["takeProfit"] = {"triggerPrice": self.client.price_to_precision(symbol, take_profit)}
        order = self.client.create_order(symbol, "market", side, contracts, None, params)
        units = contracts * cs
        price = float(order.get("average") or order.get("price") or ref_price)
        fee = abs(float((order.get("fee") or {}).get("cost") or units * price * self.fee_rate))
        return Fill(units if qty > 0 else -units, price, fee)
