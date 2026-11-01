"""Nightly LightGBM retraining with champion/challenger promotion.

1. Out-of-sample probabilities via walk-forward.
2. Tune threshold + risk params on the tune period.
3. Score the challenger and the current champion on the untouched holdout.
4. Promote only if the challenger beats the champion's risk-adjusted score (see registry).
5. Store every version (params, metrics, model file) so progress is visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradebot import registry
from tradebot.config import Settings
from tradebot.ml.model import ModelConfig, dump_model, fit, training_frame
from tradebot.ml.walkforward import (
    WalkForwardConfig,
    evaluate,
    holdout_start,
    model_signals,
    tune,
    walk_forward_probs,
)
from tradebot.store import Store

log = logging.getLogger(__name__)

AGENT_ID = "lgbm"


@dataclass
class RetrainResult:
    version: int
    promoted: bool
    challenger: dict[str, Any]
    champion: dict[str, Any]
    champion_id: str


def retrain(
    settings: Settings, store: Store, dfs: dict[str, pd.DataFrame],
    wf: WalkForwardConfig | None = None, cfg: ModelConfig | None = None,
) -> RetrainResult:
    wf = wf or WalkForwardConfig(initial_equity=settings.backtest.initial_equity)
    cfg = cfg or ModelConfig()
    ho_start = holdout_start(dfs, wf)
    end = max(df.index[-1] for df in dfs.values())

    probs = walk_forward_probs(dfs, cfg, wf)
    tuned_cfg, tuned_risk, tune_score = tune(dfs, probs, cfg, settings.risk, settings.costs, wf, ho_start)
    challenger = evaluate(
        dfs, dict(model_signals(probs, tuned_cfg)), tuned_risk, settings.costs, wf, ho_start)
    challenger["tune_score"] = tune_score

    # A LightGBM champion is re-evaluated through the same walk-forward (no leakage).
    champ_signals = None
    champ_row = registry.champion_version(store, AGENT_ID)
    if registry.champion_id(store) == AGENT_ID and champ_row is not None:
        champ_cfg = ModelConfig.from_dict(champ_row["params"]["model"])
        champ_probs = probs if _same_model(champ_cfg, cfg) else walk_forward_probs(dfs, champ_cfg, wf)
        champ_signals = model_signals(champ_probs, champ_cfg)
    champ_id, champion = registry.evaluate_champion(settings, store, dfs, wf, ho_start, champ_signals)

    registry.record_baselines(store, settings, dfs, wf, ho_start, end)

    # Final model for live trading: refit on the most recent training window.
    recent = {s: df[df.index >= end - pd.DateOffset(months=wf.train_months)] for s, df in dfs.items()}
    x, y = training_frame(recent, tuned_cfg.horizon)
    booster = fit(x, y, tuned_cfg)
    version = registry.next_version(store, AGENT_ID)
    params = {"model": tuned_cfg.to_dict(), "risk": tuned_risk.model_dump()}
    promoted = registry.record_version(
        store, settings, agent_id=AGENT_ID, kind="ml", version=version,
        blob=dump_model(booster, tuned_cfg, {"risk": tuned_risk.model_dump()}), params=params,
        challenger=challenger, champion=champion, champ_id=champ_id, holdout=(ho_start, end),
        train_range=(x.index.min(), x.index.max()), min_trades=wf.min_trades,
    )
    registry.trainer_status(store, "nightly", {"agent": AGENT_ID, "version": version, "promoted": promoted})
    return RetrainResult(version, promoted, challenger, champion, champ_id)


def _same_model(a: ModelConfig, b: ModelConfig) -> bool:
    return a.horizon == b.horizon and a.lgbm == b.lgbm and a.num_boost_round == b.num_boost_round


