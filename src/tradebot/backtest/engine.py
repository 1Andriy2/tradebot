"""Bar-by-bar futures backtester (linear USDT contracts, long and short).

Timing: the signal of bar i is computed from data up to bar i's close; the resulting
order fills at bar i+1's open with slippage. Stops are checked intrabar on bar highs/lows
(stop-loss assumed first if both levels are touched in one bar).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradebot import indicators as ind
from tradebot.config import CostsConfig, RiskParams
from tradebot.data.funding import funding_per_bar
from tradebot.paper import PaperBook
from tradebot.policy import DecisionPolicy
from tradebot.risk import HARD_LIMITS, HardLimits, RiskManager
from tradebot.simulate import Bar, step_bar
from tradebot.strategies.base import Strategy


@dataclass
class BacktestResult:
    label: str
    equity: pd.Series
    fills: pd.DataFrame
    trades: pd.DataFrame
    exposure: pd.Series
    risk_events: list[dict[str, object]] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    costs: CostsConfig,
    risk_params: RiskParams,
    initial_equity: float = 10_000.0,
    limits: HardLimits = HARD_LIMITS,
    signal: pd.Series | None = None,
    funding: pd.Series | None = None,
) -> BacktestResult:
    """Run one strategy over OHLCV ``df``. ``signal`` overrides ``strategy.signal(df)``;
    ``funding`` (rates at funding timestamps) charges perpetual funding on open positions."""
    if len(df) < 2:
        raise ValueError("need at least two bars")
    sig = (signal if signal is not None else strategy.signal(df)).reindex(df.index).fillna(0.0)
    sig_v = sig.clip(-1, 1).to_numpy(float)
    atr_v = ind.atr(df, risk_params.atr_period).to_numpy(float)
    o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = df.index
    days = pd.DatetimeIndex(idx).date
    fund_v = funding_per_bar(pd.DatetimeIndex(idx), funding).to_numpy(float)

    risk = RiskManager(risk_params, limits)
    policy = DecisionPolicy(risk)
    state = policy.new_state(initial_equity)
    book = PaperBook(initial_equity, costs)

    equity = np.empty(len(df))
    exposure = np.empty(len(df))
    n = len(df)
    for i in range(n):
        bar = Bar(idx[i], days[i], o[i], h[i], lo[i], c[i], float(sig_v[i]),
                  float(atr_v[i - 1]) if i > 0 else float("nan"), float(fund_v[i]))
        step_bar(book, state, policy, bar, strategy.uses_stops, allow_order=i + 1 < n)
        eq = book.equity(c[i])
        equity[i] = eq
        exposure[i] = book.units * c[i] / eq if eq > 0 else 0.0

    if book.units != 0:
        book.fill(idx[-1], 0.0, c[-1], "end_of_test")
        equity[-1] = book.equity(c[-1])

    return BacktestResult(
        label=strategy.label,
        equity=pd.Series(equity, index=idx, name="equity"),
        fills=pd.DataFrame(book.fills),
        trades=pd.DataFrame(book.trades),
        exposure=pd.Series(exposure, index=idx, name="exposure"),
        risk_events=state.risk.events,
    )
