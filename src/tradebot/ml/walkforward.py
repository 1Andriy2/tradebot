"""Walk-forward training, parameter tuning and out-of-sample evaluation.

Timeline (all out-of-sample predictions come from models trained only on earlier data):

    |---- train (rolling window) ----| purge |-- test fold --|  ...repeated every `test_months`
    OOS predictions: [ tune period ..................... | holdout (last `holdout_months`) ]

Signal threshold and risk parameters are tuned on the tune period only; promotion decisions
use the holdout, which no tuning step has seen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from tradebot.backtest.engine import run_backtest
from tradebot.config import CostsConfig, RiskParams
from tradebot.metrics import compute_metrics
from tradebot.ml.features import build_features, make_label
from tradebot.ml.model import ModelConfig, ModelStrategy, fit, probs_to_signal
from tradebot.risk import HARD_LIMITS
from tradebot.strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    first_test: str = "2022-01-01"
    train_months: int = 24
    test_months: int = 3
    holdout_months: int = 9
    min_trades: int = 30          # across symbols, in the evaluated period
    n_trials: int = 100
    initial_equity: float = 10_000.0


def walk_forward_probs(
    dfs: dict[str, pd.DataFrame], cfg: ModelConfig, wf: WalkForwardConfig
) -> dict[str, pd.Series]:
    """Out-of-sample P(up) per symbol, from a pooled model refit every fold."""
    feats = {s: build_features(df) for s, df in dfs.items()}
    labels = {s: make_label(df, cfg.horizon) for s, df in dfs.items()}
    end = max(df.index[-1] for df in dfs.values())
    probs = {s: np.full(len(df), np.nan) for s, df in dfs.items()}
    bar = min(pd.Timedelta(pd.Series(df.index).diff().median()) for df in dfs.values())
    purge = bar * cfg.horizon  # labels of the last training rows must not see the test period

    test_start = pd.Timestamp(wf.first_test, tz="UTC")
    while test_start <= end:
        test_end = test_start + pd.DateOffset(months=wf.test_months)
        train_start = test_start - pd.DateOffset(months=wf.train_months)
        xs, ys = [], []
        for s in dfs:
            x, y = feats[s], labels[s]
            m = (x.index >= train_start) & (x.index < test_start - purge)
            ok = m & x.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
            xs.append(x[ok])
            ys.append(y[ok])
        x_train, y_train = pd.concat(xs), pd.concat(ys)
        if len(x_train) > 1000:
            booster = fit(x_train, y_train, cfg)
            for s in dfs:
                x = feats[s]
                m = (x.index >= test_start) & (x.index < test_end) & x.notna().all(axis=1).to_numpy()
                if m.any():
                    probs[s][m] = np.asarray(booster.predict(x[m]), dtype=float)
            log.info("fold %s..%s trained on %d rows", test_start.date(), test_end.date(), len(x_train))
        test_start = test_end
    return {s: pd.Series(p, index=dfs[s].index) for s, p in probs.items()}


def holdout_start(dfs: dict[str, pd.DataFrame], wf: WalkForwardConfig) -> pd.Timestamp:
    end = max(df.index[-1] for df in dfs.values())
    return pd.Timestamp(end - pd.DateOffset(months=wf.holdout_months))


def _window(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    if start is not None:
        m &= df.index >= start
    if end is not None:
        m &= df.index < end
    return df[m]


class _SignalStrategy(Strategy):
    """Wraps a precomputed signal so it can be backtested like any strategy."""

    name = "lgbm"

    def __init__(self, signal: pd.Series, **params: Any) -> None:
        super().__init__(**params)
        self._signal = signal

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return self._signal.reindex(df.index)


def evaluate(
    dfs: dict[str, pd.DataFrame], signal_for: dict[str, Strategy | pd.Series], risk: RiskParams,
    costs: CostsConfig, wf: WalkForwardConfig,
    start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Backtest per symbol on [start, end); returns per-symbol metrics and the aggregate score."""
    per: dict[str, dict[str, float]] = {}
    for s, df in dfs.items():
        win = _window(df, start, end)
        src = signal_for[s]
        strat = src if isinstance(src, Strategy) else _SignalStrategy(src)
        res = run_backtest(win, strat, costs, risk, wf.initial_equity / len(dfs))
        per[s] = compute_metrics(res.equity, res.trades, win["close"])
    trades = sum(m["num_trades"] for m in per.values())
    score = float(np.mean([m["sortino"] for m in per.values()]))
    return {"per_symbol": per, "score": score, "num_trades": trades,
            "max_drawdown": max(m["max_drawdown"] for m in per.values()),
            "total_return": float(np.mean([m["total_return"] for m in per.values()])),
            "sharpe": float(np.mean([m["sharpe"] for m in per.values()]))}


def tune(
    dfs: dict[str, pd.DataFrame], probs: dict[str, pd.Series], cfg: ModelConfig,
    base_risk: RiskParams, costs: CostsConfig, wf: WalkForwardConfig, tune_end: pd.Timestamp,
) -> tuple[ModelConfig, RiskParams, float]:
    """Optuna search for signal threshold + risk params on the tune period only."""
    tune_start = pd.Timestamp(wf.first_test, tz="UTC")
    lim = HARD_LIMITS

    def objective(trial: optuna.Trial) -> float:
        thr = trial.suggest_float("threshold", 0.0, 0.12)
        smooth = trial.suggest_int("smooth", 1, 12)
        risk = RiskParams(
            position_frac=trial.suggest_float("position_frac", 0.05, lim.max_position_frac),
            leverage=trial.suggest_float("leverage", 1.0, lim.max_leverage),
            stop_atr_mult=trial.suggest_float("stop_atr_mult", 1.0, 8.0),
            take_atr_mult=trial.suggest_float("take_atr_mult", 1.0, 12.0),
            atr_period=base_risk.atr_period,
        )
        sig = {s: probs_to_signal(p, thr, smooth) for s, p in probs.items()}
        r = evaluate(dfs, dict(sig), risk, costs, wf, tune_start, tune_end)
        if r["num_trades"] < wf.min_trades:
            return -10.0
        return float(r["score"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
    study.optimize(objective, n_trials=wf.n_trials)
    p = study.best_params
    tuned = ModelConfig(**{**cfg.to_dict(), "threshold": p["threshold"], "smooth": p["smooth"]})
    risk = RiskParams(position_frac=p["position_frac"], leverage=p["leverage"],
                      stop_atr_mult=p["stop_atr_mult"], take_atr_mult=p["take_atr_mult"],
                      atr_period=base_risk.atr_period)
    return tuned, risk, float(study.best_value)


def model_signals(probs: dict[str, pd.Series], cfg: ModelConfig) -> dict[str, pd.Series]:
    return {s: probs_to_signal(p, cfg.threshold, cfg.smooth) for s, p in probs.items()}


__all__ = ["WalkForwardConfig", "walk_forward_probs", "tune", "evaluate", "holdout_start",
           "model_signals", "ModelStrategy"]
