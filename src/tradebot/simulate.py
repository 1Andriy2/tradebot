"""One simulated bar: used by the backtester and by live shadow (paper) agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradebot.paper import PaperBook
from tradebot.policy import Decision, DecisionPolicy, PolicyState, check_stops


@dataclass(frozen=True)
class Bar:
    ts: pd.Timestamp
    day: date
    open: float
    high: float
    low: float
    close: float
    signal: float      # strategy output at this bar's close
    prev_atr: float    # ATR at the previous bar's close (known before this bar)


def step_bar(
    book: PaperBook, state: PolicyState, policy: DecisionPolicy, bar: Bar,
    uses_stops: bool = True, allow_order: bool = True,
) -> Decision | None:
    """Fill the pending order at the open, check stops intrabar, decide at the close.

    Returns the decision taken at this bar's close (it fills at the next bar's open).
    """
    pending = state.extra.pop("pending", None)
    if pending is not None:
        book.fill(bar.ts, pending["target"], bar.open, pending["reason"])
        state.sl = state.tp = None  # re-derived from the new average entry
    if uses_stops and book.units != 0 and state.sl is None and state.tp is None:
        state.sl, state.tp = policy.risk.stop_levels(np.sign(book.units), book.avg_entry, bar.prev_atr)

    hit = check_stops(book.units, state.sl, state.tp, bar.open, bar.high, bar.low)
    if hit is not None:
        book.fill(bar.ts, 0.0, hit[0], hit[1])
        policy.on_stopped(state)

    decision = policy.decide(state, bar.day, book.equity(bar.close), bar.signal, bar.close, book.units)
    if decision is not None and allow_order:
        state.extra["pending"] = {"target": decision.target_units, "reason": decision.reason}
    return decision
