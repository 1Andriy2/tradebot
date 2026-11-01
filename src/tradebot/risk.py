"""Risk management shared by the backtester and the live (demo) trader.

Position size convention: ``position_frac`` is the share of equity committed as margin and
``leverage`` multiplies it, so notional = signal * position_frac * leverage * equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from tradebot.config import RiskParams


@dataclass(frozen=True)
class HardLimits:
    """Non-negotiable caps. Learned parameters are clipped to these; they are never tuned."""

    max_position_frac: float = 0.25
    max_leverage: float = 2.0
    daily_loss_limit: float = 0.05
    kill_switch_drawdown: float = 0.20


HARD_LIMITS = HardLimits()

KILL_SWITCH = "kill_switch"
DAILY_LOSS_LIMIT = "daily_loss_limit"


@dataclass
class RiskState:
    peak_equity: float
    day: date | None = None
    day_start_equity: float = 0.0
    killed: bool = False
    halted_day: date | None = None
    events: list[dict[str, object]] = field(default_factory=list)


class RiskManager:
    def __init__(self, params: RiskParams, limits: HardLimits = HARD_LIMITS) -> None:
        self.params = params
        self.limits = limits
        self.position_frac = min(params.position_frac, limits.max_position_frac)
        self.leverage = min(params.leverage, limits.max_leverage)

    def new_state(self, equity: float) -> RiskState:
        return RiskState(peak_equity=equity, day_start_equity=equity)

    def check(self, state: RiskState, day: date, equity: float) -> str | None:
        """Update drawdown/daily trackers; return a blocking reason or None."""
        if state.day != day:
            state.day = day
            state.day_start_equity = equity
        state.peak_equity = max(state.peak_equity, equity)

        if state.killed:
            return KILL_SWITCH
        drawdown = 1 - equity / state.peak_equity
        if drawdown >= self.limits.kill_switch_drawdown:
            state.killed = True
            state.events.append({"type": KILL_SWITCH, "day": day, "equity": equity,
                                 "drawdown": drawdown})
            return KILL_SWITCH

        if state.halted_day == day:
            return DAILY_LOSS_LIMIT
        day_loss = 1 - equity / state.day_start_equity
        if day_loss >= self.limits.daily_loss_limit:
            state.halted_day = day
            state.events.append({"type": DAILY_LOSS_LIMIT, "day": day, "equity": equity,
                                 "day_loss": day_loss})
            return DAILY_LOSS_LIMIT
        return None

    def target_units(self, signal: float, equity: float, price: float) -> float:
        signal = max(-1.0, min(1.0, signal))
        return signal * self.position_frac * self.leverage * equity / price

    def stop_levels(self, side: float, entry: float, atr: float) -> tuple[float | None, float | None]:
        """Stop-loss and take-profit prices for a position of the given side (+1/-1)."""
        if atr != atr or atr <= 0:  # NaN or non-positive
            return None, None
        sl = tp = None
        if self.params.stop_atr_mult:
            sl = entry - side * self.params.stop_atr_mult * atr
        if self.params.take_atr_mult:
            tp = entry + side * self.params.take_atr_mult * atr
        return sl, tp
