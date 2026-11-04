"""Walk-forward strategy research.

For every (strategy family, symbol, timeframe):
  for each test fold, pick the family's parameters that scored best on the preceding
  training window (fitted strategies are also fitted there), then trade the next fold with
  them. The out-of-sample folds are stitched into one backtest — with fees, slippage and
  funding — so every number reported is out-of-sample.

Besides the usual metrics, each result carries:
  * fold_win_rate  — share of test folds that made money (does the edge repeat?);
  * dsr            — Deflated Sharpe Ratio: probability that the true Sharpe is > 0 after
                     correcting for how many configurations were tried (Bailey & López de Prado);
  * robustness     — overfitting, significance and stress statistics, see research/robustness.py.

These numbers only inform: nothing here changes which agent trades.
"""

from __future__ import annotations

import logging
import math
import uuid
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tradebot.backtest.engine import run_backtest
from tradebot.config import CostsConfig, RiskParams
from tradebot.metrics import compute_metrics
from tradebot.ml.features import resample_ohlcv
from tradebot.research import robustness as rb
from tradebot.research.patterns import bh_qvalues
from tradebot.risk import HARD_LIMITS
from tradebot.strategies.base import Strategy
from tradebot.strategies.library import FAMILIES, make

log = logging.getLogger(__name__)

EULER_GAMMA = 0.5772156649
TF_RULES = {"15m": None, "1h": "1h", "4h": "4h"}


@dataclass(frozen=True)
class ResearchConfig:
    timeframes: tuple[str, ...] = ("15m", "1h", "4h")
    families: tuple[str, ...] = tuple(FAMILIES)
    first_test: str = "2022-01-01"
    train_months: int = 12
    test_months: int = 3
    min_train_trades: int = 5
    initial_equity: float = 10_000.0
    workers: int = 4
    robustness: bool = True
    mc_sims: int = 2000
    mc_mean_block: int = 10
    ruin_drawdown: float = HARD_LIMITS.kill_switch_drawdown


@dataclass
class Task:
    family: str
    symbol: str
    timeframe: str
    df: pd.DataFrame
    funding: pd.Series | None
    cfg: ResearchConfig
    costs: CostsConfig
    risk: RiskParams
    extra: dict[str, Any] = field(default_factory=dict)


class _Stitched(Strategy):
    """Backtests a precomputed out-of-sample signal with the family's stop behaviour."""

    name = "oos"

    def __init__(self, signal: pd.Series, uses_stops: bool) -> None:
        super().__init__()
        self._signal = signal
        self.uses_stops = uses_stops

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return self._signal.reindex(df.index)


def to_timeframe(df15: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = TF_RULES[timeframe]
    return df15 if rule is None else resample_ohlcv(df15, rule)


def _score(m: dict[str, float], min_trades: int) -> float:
    return m["sortino"] if m["num_trades"] >= min_trades else -math.inf


def _backtest_params(t: Task, params: dict[str, Any], fit_on: pd.DataFrame, win: pd.DataFrame) -> pd.Series:
    """Equity of one fixed parameter set over the research window, indicators warmed up."""
    strat = make(t.family, params)
    strat.fit(fit_on)
    sig = strat.signal(t.df).reindex(win.index)
    return run_backtest(win, _Stitched(sig, strat.uses_stops), t.costs, t.risk, t.cfg.initial_equity,
                        funding=t.funding).equity


def _sortino_of(equity: pd.Series) -> float:
    return compute_metrics(equity, pd.DataFrame())["sortino"]


def _robustness(t: Task, win: pd.DataFrame, oos: pd.Series, equity: pd.Series, sortino: float,
                folds: list[dict[str, Any]]) -> tuple[dict[str, float | None], pd.DataFrame]:
    """Statistics for one task, plus its days x configurations return matrix for the run-level PBO.

    Fitted families are fitted once on the data before the window, so the matrix stays
    out-of-sample. Latest-parameter runs measure stability, they are not performance claims.
    """
    cfg = t.cfg
    _, grid = FAMILIES[t.family]
    fit_start = win.index[0] - pd.DateOffset(months=cfg.train_months)
    fit_on = t.df[(t.df.index >= fit_start) & (t.df.index < win.index[0])]
    daily = rb.daily_returns(equity)
    columns = {f"{t.family}|{i}": rb.daily_returns(_backtest_params(t, p, fit_on, win))
               for i, p in enumerate(grid)}
    frame = pd.DataFrame(columns).fillna(0.0)

    out: dict[str, float | None] = {
        "ulcer_index": rb.ulcer_index(equity.resample("1D").last().ffill()),
        "cvar_5": rb.cvar(daily),
        "tail_ratio": rb.tail_ratio(daily),
        "pbo": rb.pbo(frame.to_numpy()),
        **rb.cpcv_stats(frame.to_numpy()),
    }
    rng = np.random.default_rng(zlib.crc32(f"{t.family}|{t.symbol}|{t.timeframe}".encode()))
    out["signal_p"] = rb.rule_significance(oos, win["close"], cfg.mc_sims, cfg.mc_mean_block, rng)
    if len(daily) >= rb.MIN_DAYS and daily.std() > 0:  # a strategy that never traded has no spread
        sr = float(daily.mean() / daily.std())
        out["psr"] = deflated_sharpe(sr, [sr], 1, float(len(daily)), float(stats.skew(daily)),
                                     float(stats.kurtosis(daily, fisher=False)))
        idx = rb.stationary_bootstrap_indices(len(daily), cfg.mc_sims, cfg.mc_mean_block, rng)
        out.update(rb.monte_carlo(daily.to_numpy(), idx, cfg.ruin_drawdown))

    latest = folds[-1]["params"] if folds else {}
    moved = [_sortino_of(_backtest_params(t, p, fit_on, win)) for p in rb.perturbed_params(latest)]
    base = _sortino_of(_backtest_params(t, latest, fit_on, win)) if moved else 0.0
    out["param_stability"] = rb.param_stability(base, moved)
    train_scores = [f["train_score"] for f in folds if f["train_score"] is not None]
    train_mean = float(np.mean(train_scores)) if train_scores else 0.0
    out["wf_retention"] = sortino / train_mean if train_mean > 0 else None
    return out, frame


def run_task(t: Task) -> dict[str, Any]:
    df, cfg = t.df, t.cfg
    _, grid = FAMILIES[t.family]
    end = df.index[-1]
    first = pd.Timestamp(cfg.first_test, tz="UTC")
    oos = pd.Series(0.0, index=df.index)
    folds: list[dict[str, Any]] = []
    uses_stops = make(t.family, grid[0]).uses_stops

    test_start = first
    while test_start < end:
        test_end = test_start + pd.DateOffset(months=cfg.test_months)
        train = df[(df.index >= test_start - pd.DateOffset(months=cfg.train_months))
                   & (df.index < test_start)]
        if len(train) < 100:
            test_start = test_end
            continue
        best_params, best_score = grid[0], -math.inf
        if len(grid) > 1:
            for params in grid:
                strat = make(t.family, params)
                strat.fit(train)
                r = run_backtest(train, strat, t.costs, t.risk, cfg.initial_equity, funding=t.funding)
                s = _score(compute_metrics(r.equity, r.trades), cfg.min_train_trades)
                if s > best_score:
                    best_params, best_score = params, s
        strat = make(t.family, best_params)
        strat.fit(train)
        history = df[df.index < test_end]  # causal: indicators warm up on earlier data
        sig = strat.signal(history)
        mask = (df.index >= test_start) & (df.index < test_end)
        oos[mask] = sig.reindex(df.index[mask]).to_numpy()
        folds.append({"test_start": test_start.isoformat(), "params": best_params,
                      "train_score": None if math.isinf(best_score) else best_score})
        test_start = test_end

    win = df[df.index >= first]
    res = run_backtest(win, _Stitched(oos, uses_stops), t.costs, t.risk, cfg.initial_equity,
                       funding=t.funding)
    m: dict[str, Any] = dict(compute_metrics(res.equity, res.trades, win["close"]))

    # Did the edge repeat? Return of each test fold, from the stitched equity curve.
    fold_rets = []
    for f in folds:
        a = pd.Timestamp(f["test_start"])
        b = a + pd.DateOffset(months=cfg.test_months)
        seg = res.equity[(res.equity.index >= a) & (res.equity.index < b)]
        if len(seg) > 1:
            fold_rets.append(float(seg.iloc[-1] / seg.iloc[0] - 1))
    rets = res.equity.pct_change().dropna()
    funding_paid = float(res.trades["funding"].sum()) if len(res.trades) else 0.0
    fees_paid = float(res.trades["fees"].sum()) if len(res.trades) else 0.0
    m.update({
        "fold_win_rate": float(np.mean([r > 0 for r in fold_rets])) if fold_rets else 0.0,
        "n_folds": float(len(fold_rets)),
        "funding_paid": funding_paid,
        "fees_paid": fees_paid,
        "sr_period": float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0,
        "skew": float(stats.skew(rets)) if len(rets) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(rets, fisher=False)) if len(rets) > 3 else 3.0,
        "n_obs": float(len(rets)),
        "grid_size": float(len(grid)),
    })
    result: dict[str, Any] = {
        "family": t.family, "symbol": t.symbol, "timeframe": t.timeframe,
        "metrics": m, "params": {"folds": folds, "latest": folds[-1]["params"] if folds else {}},
        "fold_returns": fold_rets,
    }
    if cfg.robustness:
        extra, result["_returns"] = _robustness(t, win, oos[oos.index >= first], res.equity,
                                                m["sortino"], folds)
        m.update(extra)
    return result


def deflated_sharpe(sr: float, sr_trials: list[float], n_trials: int, n_obs: float,
                    skew: float, kurtosis: float) -> float:
    """P(true Sharpe > 0) after selecting the best of ``n_trials`` (per-period Sharpe units)."""
    if n_obs < 3 or n_trials < 1:
        return 0.0
    var = float(np.var(sr_trials)) if len(sr_trials) > 1 else 0.0
    if n_trials > 1 and var > 0:
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
        sr0 = math.sqrt(var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurtosis - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom))


def _peer_profit_share(
    row: dict[str, Any], results: list[dict[str, Any]], same: str, other: str
) -> float | None:
    """Share of the same family's results on other values of ``other`` that made money."""
    peers = [r for r in results if r["family"] == row["family"] and r[same] == row[same]
             and r[other] != row[other]]
    return float(np.mean([r["metrics"]["total_return"] > 0 for r in peers])) if peers else None


def _add_run_level_robustness(results: list[dict[str, Any]]) -> None:
    """Statistics that need the whole run: multiple testing, peers, and PBO across families."""
    frames = [r.pop("_returns") for r in results]
    universe: dict[tuple[str, str], float | None] = {}
    for key in {(r["symbol"], r["timeframe"]) for r in results}:
        group = [f for r, f in zip(results, frames, strict=True) if (r["symbol"], r["timeframe"]) == key]
        universe[key] = rb.pbo(pd.concat(group, axis=1).fillna(0.0).to_numpy())

    tested = [r for r in results if r["metrics"].get("signal_p") is not None]
    if tested:
        q_values = bh_qvalues(np.array([r["metrics"]["signal_p"] for r in tested]))
        for r, q in zip(tested, q_values, strict=True):
            r["metrics"]["signal_q"] = float(q)
    for r in results:
        m = r["metrics"]
        m["pbo_universe"] = universe[(r["symbol"], r["timeframe"])]
        m["peer_pair_profitable"] = _peer_profit_share(r, results, "timeframe", "symbol")
        m["peer_tf_profitable"] = _peer_profit_share(r, results, "symbol", "timeframe")
        m.update(rb.verdict(m))


def run_research(
    dfs15: dict[str, pd.DataFrame], funding: dict[str, pd.Series | None], costs: CostsConfig,
    risk: RiskParams, cfg: ResearchConfig | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    cfg = cfg or ResearchConfig()
    frames = {(s, tf): to_timeframe(df, tf) for s, df in dfs15.items() for tf in cfg.timeframes}
    tasks = [Task(fam, s, tf, frames[(s, tf)], funding.get(s), cfg, costs, risk)
             for fam in cfg.families for (s, tf) in frames]
    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            results = list(pool.map(run_task, tasks))
    else:
        results = [run_task(t) for t in tasks]

    n_trials = int(sum(r["metrics"]["grid_size"] for r in results))
    srs = [r["metrics"]["sr_period"] for r in results]
    for r in results:
        m = r["metrics"]
        m["dsr"] = deflated_sharpe(m["sr_period"], srs, n_trials, m["n_obs"], m["skew"], m["kurtosis"])
        m["n_trials"] = float(n_trials)
    if cfg.robustness:
        _add_run_level_robustness(results)
    return str(uuid.uuid4()), results
