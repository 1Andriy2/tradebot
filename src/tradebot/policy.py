"""Turns per-bar signals into position targets. Shared by the backtester and live trading
so that both follow exactly the same rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from tradebot.risk import RiskManager, RiskState

EPS = 1e-12


@dataclass
class PolicyState:
    risk: RiskState
    prev_sig: float = 0.0
    suppressed_sig: float | None = None  # after a stop, wait for the signal to change
    sl: float | None = None
    tp: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        r = d["risk"]
        for key in ("day", "halted_day"):
            r[key] = r[key].isoformat() if r[key] else None
        r["events"] = []  # events are persisted separately
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyState:
        r = dict(d["risk"])
        for key in ("day", "halted_day"):
            r[key] = date.fromisoformat(r[key]) if r.get(key) else None
        r["events"] = []
        return cls(risk=RiskState(**r), prev_sig=d["prev_sig"], suppressed_sig=d["suppressed_sig"],
                   sl=d.get("sl"), tp=d.get("tp"), extra=d.get("extra", {}))


@dataclass(frozen=True)
class Decision:
    target_units: float
    reason: str


class DecisionPolicy:
    def __init__(self, risk: RiskManager) -> None:
        self.risk = risk

    def new_state(self, equity: float) -> PolicyState:
        return PolicyState(risk=self.risk.new_state(equity))

    def on_stopped(self, state: PolicyState) -> None:
        state.suppressed_sig = state.prev_sig
        state.sl = state.tp = None

    def decide(
        self, state: PolicyState, day: date, equity: float, signal: float, price: float, units: float
    ) -> Decision | None:
        """Called at a bar close. Returns a new target, or None to keep the current position."""
        signal = max(-1.0, min(1.0, signal))
        block = self.risk.check(state.risk, day, equity)
        if state.suppressed_sig is not None and signal != state.suppressed_sig:
            state.suppressed_sig = None

        target, reason = units, "signal"
        if block:
            target, reason = 0.0, block
        elif state.suppressed_sig is None and (
            signal != state.prev_sig or (abs(units) < EPS and signal != 0)
        ):
            target = self.risk.target_units(signal, equity, price)
        state.prev_sig = signal
        if abs(target - units) <= EPS:
            return None
        return Decision(target, reason)


def check_stops(
    units: float, sl: float | None, tp: float | None, open_: float, high: float, low: float
) -> tuple[float, str] | None:
    """Intrabar stop/take check for one bar. Stop-loss wins if both are touched.
    Returns (fill reference price, reason) or None."""
    if abs(units) < EPS:
        return None
    long = units > 0
    hit_sl = sl is not None and (low <= sl if long else high >= sl)
    hit_tp = tp is not None and (high >= tp if long else low <= tp)
    if not (hit_sl or hit_tp):
        return None
    level = sl if hit_sl else tp
    assert level is not None
    gapped = (open_ <= level) if (long == hit_sl) else (open_ >= level)
    return (open_ if gapped else level), ("stop_loss" if hit_sl else "take_profit")
