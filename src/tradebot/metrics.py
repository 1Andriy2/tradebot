"""Performance metrics computed from an equity curve and a trade list."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 365 * 24 * 60


def periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    step = pd.Series(index).diff().median()
    return MINUTES_PER_YEAR / (step.total_seconds() / 60)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough loss as a positive fraction."""
    peak = equity.cummax()
    return float((1 - equity / peak).max()) if len(equity) else 0.0


def sharpe(returns: pd.Series, ppy: float) -> float:
    std = returns.std(ddof=0)
    if not std or math.isnan(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(ppy))


def sortino(returns: pd.Series, ppy: float) -> float:
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    if not downside or math.isnan(downside):
        return 0.0
    return float(returns.mean() / downside * math.sqrt(ppy))


def compute_metrics(
    equity: pd.Series, trades: pd.DataFrame, close: pd.Series | None = None
) -> dict[str, float]:
    """All dashboard metrics. ``close`` enables the comparison against raw Buy & Hold."""
    rets = equity.pct_change().dropna()
    ppy = periods_per_year(pd.DatetimeIndex(equity.index))
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    n = len(trades)
    pnl = trades["pnl"] if n else pd.Series(dtype=float)
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = math.inf if gross_win > 0 else 0.0

    out = {
        "total_return": total_return,
        "sharpe": sharpe(rets, ppy),
        "sortino": sortino(rets, ppy),
        "max_drawdown": max_drawdown(equity),
        "win_rate": float((pnl > 0).mean()) if n else 0.0,
        "profit_factor": profit_factor,
        "num_trades": float(n),
    }
    if close is not None:
        bh = float(close.iloc[-1] / close.iloc[0] - 1)
        out["buy_hold_return"] = bh
        out["excess_vs_buy_hold"] = total_return - bh
    return out
