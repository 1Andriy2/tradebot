"""Statistics that ask whether a research result is real or a lucky pick.

Pure functions on returns and signals; lab.py runs the backtests that feed them. Everything
works on daily returns so that 15m, 1h and 4h results are comparable and cheap to resample.

Methods and where they come from (all reimplemented here, no third-party code is imported):
  * Ulcer index, CVaR, tail ratio: the formulas QuantStats reports.
  * Combinatorial purged CV: López de Prado, "Advances in Financial Machine Learning", ch. 12.
    skfolio's ``CombinatorialPurgedCV`` was the reference for fold and path bookkeeping.
  * Probability of backtest overfitting (CSCV): Bailey, Borwein, López de Prado, Zhu (2017).
  * Rule significance on detrended returns with a stationary bootstrap (Politis & Romano 1994):
    the test design of Jesse's ``rule_significance_testing``.
  * Parameter sensitivity and peer checks: DaviddTech's overfitting checklist.
"""

from __future__ import annotations

import itertools
import math
import operator
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

MIN_DAYS = 60  # below this the resampling statistics are noise
DAYS_PER_YEAR = 365

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int64]


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple daily returns from an equity curve of any bar size."""
    return equity.resample("1D").last().ffill().pct_change().dropna()


def ulcer_index(equity: pd.Series) -> float:
    """Root mean square drawdown: punishes long, deep underwater periods."""
    if equity.empty:
        return 0.0
    drawdown = equity / equity.cummax() - 1
    return float(np.sqrt((drawdown**2).mean()))


def cvar(returns: pd.Series, level: float = 0.05) -> float:
    """Mean of the worst ``level`` share of returns (a negative number means a loss)."""
    if returns.empty:
        return 0.0
    cutoff = returns.quantile(level)
    return float(returns[returns <= cutoff].mean())


def tail_ratio(returns: pd.Series) -> float | None:
    """Size of the right tail over the left tail, ignoring flat days."""
    active = returns[returns != 0]
    if active.empty:
        return None
    left = abs(float(active.quantile(0.05)))
    return abs(float(active.quantile(0.95))) / left if left > 0 else None


def stationary_bootstrap_indices(n: int, n_sims: int, mean_block: int, rng: np.random.Generator) -> Ints:
    """Resampled positions with geometric block lengths, wrapping at the end of the series."""
    if n < 1 or mean_block < 1:
        raise ValueError("n and mean_block must be at least 1")
    idx = np.empty((n_sims, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_sims)
    for t in range(1, n):
        restart = rng.random(n_sims) < 1 / mean_block
        idx[:, t] = np.where(restart, rng.integers(0, n, n_sims), (idx[:, t - 1] + 1) % n)
    return idx


def rule_significance(signal: pd.Series, close: pd.Series, n_sims: int, mean_block: int,
                      rng: np.random.Generator) -> float | None:
    """One-sided p-value that the signal times the market better than chance.

    H0: expected signal x next-bar detrended return is zero. Detrending removes the reward
    for simply being long in a rising market. This looks at the raw signal: costs and stops
    are not part of the test, they are in the backtest metrics.
    """
    log_next = np.log(close).diff().shift(-1)
    rule = (signal.reindex(close.index).fillna(0.0) * (log_next - log_next.mean())).dropna()
    daily = rule.resample("1D").sum().to_numpy()
    if len(daily) < MIN_DAYS:
        return None
    observed = float(daily.mean())
    idx = stationary_bootstrap_indices(len(daily), n_sims, mean_block, rng)
    simulated = (daily - observed)[idx].mean(axis=1)
    return float((1 + np.sum(simulated >= observed)) / (n_sims + 1))


def monte_carlo(daily: Floats, idx: Ints, ruin_drawdown: float) -> dict[str, float]:
    """Spread of outcomes if the same days had arrived in another order, in blocks."""
    paths = np.cumprod(1 + daily[idx], axis=1)
    peaks = np.maximum(np.maximum.accumulate(paths, axis=1), 1.0)
    drawdowns = (1 - paths / peaks).max(axis=1)
    totals = paths[:, -1] - 1
    return {
        "mc_return_p05": float(np.quantile(totals, 0.05)),
        "mc_return_p50": float(np.quantile(totals, 0.5)),
        "mc_loss_prob": float(np.mean(totals <= 0)),
        "mc_drawdown_p95": float(np.quantile(drawdowns, 0.95)),
        "mc_ruin_prob": float(np.mean(drawdowns >= ruin_drawdown)),
    }


def _sharpe(returns: Floats) -> Floats:
    """Per-column daily Sharpe; flat columns score 0."""
    std = returns.std(axis=0)
    mean = returns.mean(axis=0)
    out: Floats = np.divide(mean, std, out=np.zeros_like(mean, dtype=float), where=std > 0)
    return out


def _fold_slices(n: int, n_folds: int) -> list[Ints]:
    return [np.asarray(a) for a in np.array_split(np.arange(n), n_folds)]


def pbo(matrix: Floats, n_blocks: int = 10) -> float | None:
    """Probability of backtest overfitting (CSCV).

    ``matrix`` is days x configurations. For every half of the blocks taken as in-sample, take
    the best configuration there and see where it ranks on the other half. PBO is how often
    it lands in the bottom half. Around 0.5 means the in-sample winner carries no information.
    With few configurations ranks are coarse, so read small-grid values loosely.
    """
    n, n_cfg = matrix.shape
    if n_cfg < 2 or n < MIN_DAYS or n_blocks % 2:
        return None
    blocks = _fold_slices(n, n_blocks)
    overfit = total = 0
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        chosen = set(combo)
        in_rows = np.concatenate([blocks[i] for i in combo])
        out_rows = np.concatenate([b for i, b in enumerate(blocks) if i not in chosen])
        best = int(np.argmax(_sharpe(matrix[in_rows])))
        out_sharpe = _sharpe(matrix[out_rows])
        rank = float(np.sum(out_sharpe <= out_sharpe[best]))  # 1 = worst, n_cfg = best
        overfit += rank / (n_cfg + 1) <= 0.5
        total += 1
    return overfit / total


def cpcv_splits(n: int, n_folds: int, n_test: int, purge: int) -> list[tuple[Ints, tuple[int, ...]]]:
    """Train rows and test folds for every choice of ``n_test`` folds out of ``n_folds``.

    ``purge`` rows are removed from the training set on both sides of each test fold, so
    positions and indicator windows that span a boundary do not leak into training.
    """
    if n_test < 1 or n_folds <= n_test:
        raise ValueError("need 1 <= n_test < n_folds")
    folds = _fold_slices(n, n_folds)
    splits = []
    for combo in itertools.combinations(range(n_folds), n_test):
        keep = np.ones(n, dtype=bool)
        for f in combo:
            keep[max(0, folds[f][0] - purge): folds[f][-1] + 1 + purge] = False
        splits.append((np.flatnonzero(keep), combo))
    return splits


def cpcv_paths(matrix: Floats, n_folds: int = 6, n_test: int = 2, purge: int = 5) -> list[Floats]:
    """Out-of-sample return paths: select on the train folds, trade the test folds.

    Each fold is tested in several splits; its j-th appearance goes to path j, so every path
    is a full history. One walk-forward gives one path. This gives a spread of them.
    """
    n = matrix.shape[0]
    folds = _fold_slices(n, n_folds)
    n_paths = math.comb(n_folds, n_test) * n_test // n_folds
    pieces: list[list[Floats]] = [[np.empty(0)] * n_folds for _ in range(n_paths)]
    seen = [0] * n_folds
    for train, combo in cpcv_splits(n, n_folds, n_test, purge):
        best = int(np.argmax(_sharpe(matrix[train])))
        for f in combo:
            pieces[seen[f]][f] = matrix[folds[f], best]
            seen[f] += 1
    return [np.concatenate(p) for p in pieces]


def cpcv_stats(matrix: Floats) -> dict[str, float]:
    if matrix.shape[0] < MIN_DAYS:
        return {}
    annual = math.sqrt(DAYS_PER_YEAR)
    sharpes = [float(_sharpe(p[:, None])[0]) * annual for p in cpcv_paths(matrix)]
    return {
        "cpcv_sharpe_median": float(np.median(sharpes)),
        "cpcv_sharpe_min": min(sharpes),
        "cpcv_positive_paths": float(np.mean([s > 0 for s in sharpes])),
    }


def perturbed_params(
    params: dict[str, Any], factors: tuple[float, ...] = (0.5, 0.8, 1.2, 1.5)
) -> list[dict[str, Any]]:
    """Each numeric parameter moved on its own by each factor, dropping invalid combinations."""
    out: list[dict[str, Any]] = []
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        for factor in factors:
            moved: int | float = max(2, round(value * factor)) if isinstance(value, int) else value * factor
            candidate = {**params, key: moved}
            if moved != value and candidate not in out and _valid(candidate):
                out.append(candidate)
    return out


def _valid(p: dict[str, Any]) -> bool:
    if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
        return False
    if "low" in p and "high" in p and not 0 < p["low"] < p["high"] < 100:
        return False
    return not ("adx_min" in p and p["adx_min"] >= 100)


def param_stability(base_sortino: float, moved_sortinos: list[float]) -> float | None:
    """Share of nearby parameters that keep at least half of a positive Sortino."""
    if base_sortino <= 0 or not moved_sortinos:
        return None
    return float(np.mean([s >= base_sortino / 2 for s in moved_sortinos]))


def verdict(m: dict[str, Any]) -> dict[str, float]:
    """How many of the available checks a result passes. Thresholds are rules of thumb.

    A check whose statistic is missing is left out rather than failed. ``robust`` needs a
    profit, every check passed, and at least four checks to count.
    """
    results = [
        passes(m[key], limit)
        for key, passes, limit in (
            ("dsr", operator.ge, 0.95),
            ("signal_q", operator.lt, 0.05),
            ("pbo", operator.lt, 0.5),
            ("cpcv_positive_paths", operator.ge, 0.8),
            ("param_stability", operator.ge, 0.7),
            ("peer_pair_profitable", operator.ge, 0.5),
            ("peer_tf_profitable", operator.ge, 0.5),
            ("mc_loss_prob", operator.le, 0.1),
        )
        if m.get(key) is not None
    ]
    passed = sum(results)
    enough = len(results) >= 4 and passed == len(results)
    return {"checks_passed": float(passed), "checks_total": float(len(results)),
            "robust": float(enough and m["total_return"] > 0)}
