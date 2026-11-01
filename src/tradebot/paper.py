"""Simulated futures account used by the backtester and by shadow (paper) agents."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradebot.config import CostsConfig

EPS = 1e-12


class PaperBook:
    """Tracks cash, position, fills and round-trip trades."""

    def __init__(self, equity: float, costs: CostsConfig) -> None:
        self.cash = equity
        self.units = 0.0
        self.avg_entry = 0.0
        self.costs = costs
        self.fills: list[dict[str, object]] = []
        self.trades: list[dict[str, object]] = []
        self._open: dict[str, object] | None = None

    def fill(
        self, ts: pd.Timestamp, target: float, ref_price: float, reason: str, slippage: bool = True
    ) -> None:
        """Move the position to ``target`` units. ``slippage=False`` for real (already slipped) fills."""
        qty = target - self.units
        if abs(qty) < EPS:
            return
        # A flip is executed as a close followed by an open so trades stay well defined.
        if self.units != 0 and np.sign(target) != np.sign(self.units):
            self._execute(ts, -self.units, ref_price, reason, slippage)
            qty = target
        if abs(qty) > EPS:
            self._execute(ts, qty, ref_price, reason, slippage)

    def apply_funding(self, rate: float, price: float) -> float:
        """Pay (rate > 0, long) or receive funding on the open position; returns the payment."""
        if abs(self.units) < EPS or rate == 0.0:
            return 0.0
        payment = self.units * price * rate
        self.cash -= payment
        assert self._open is not None
        self._open["cash_flow"] = float(self._open["cash_flow"]) - payment  # type: ignore[arg-type]
        self._open["funding"] = float(self._open.get("funding", 0.0)) + payment  # type: ignore[arg-type]
        return payment

    def drain(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Return and clear completed fills and trades."""
        fills, trades = self.fills, self.trades
        self.fills, self.trades = [], []
        return fills, trades

    def _execute(
        self, ts: pd.Timestamp, qty: float, ref_price: float, reason: str, slippage: bool = True
    ) -> None:
        side = np.sign(qty)
        price = ref_price * (1 + side * self.costs.slippage_rate) if slippage else ref_price
        fee = abs(qty) * price * self.costs.fee_rate
        self.cash -= qty * price + fee

        new_units = self.units + qty
        if self.units == 0:
            self._open = {"entry_time": ts, "side": "long" if qty > 0 else "short",
                          "entry_reason": reason, "cash_flow": 0.0, "fees": 0.0, "funding": 0.0,
                          "max_units": 0.0, "entry_price": price}
            self.avg_entry = price
        elif np.sign(qty) == np.sign(self.units):
            self.avg_entry = (self.avg_entry * abs(self.units) + price * abs(qty)) / abs(new_units)

        assert self._open is not None
        self._open["cash_flow"] = float(self._open["cash_flow"]) - qty * price - fee  # type: ignore[arg-type]
        self._open["fees"] = float(self._open["fees"]) + fee  # type: ignore[arg-type]
        self._open["max_units"] = max(float(self._open["max_units"]), abs(new_units))  # type: ignore[arg-type]
        self.units = 0.0 if abs(new_units) < EPS else new_units
        self.fills.append({"time": ts, "qty": qty, "price": price, "fee": fee, "reason": reason,
                           "units_after": self.units})

        if self.units == 0:
            t = self._open
            notional = float(t["max_units"]) * float(t["entry_price"])  # type: ignore[arg-type]
            self.trades.append({
                "entry_time": t["entry_time"], "exit_time": ts, "side": t["side"],
                "entry_price": t["entry_price"], "exit_price": price,
                "entry_reason": t["entry_reason"], "exit_reason": reason,
                "pnl": t["cash_flow"], "fees": t["fees"], "funding": t.get("funding", 0.0),
                "return_pct": float(t["cash_flow"]) / notional if notional else 0.0,  # type: ignore[arg-type]
            })
            self._open = None
            self.avg_entry = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def to_dict(self) -> dict[str, Any]:
        """Persistent state (completed fills/trades are drained by the caller)."""
        return {"cash": self.cash, "units": self.units, "avg_entry": self.avg_entry,
                "open": _jsonable(self._open)}

    @classmethod
    def from_dict(cls, d: dict[str, Any], costs: CostsConfig) -> PaperBook:
        book = cls(d["cash"], costs)
        book.units, book.avg_entry = d["units"], d["avg_entry"]
        if d.get("open"):
            book._open = dict(d["open"])
            book._open["entry_time"] = pd.Timestamp(str(book._open["entry_time"]))
        return book


def _jsonable(d: dict[str, object] | None) -> dict[str, object] | None:
    if d is None:
        return None
    return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}
