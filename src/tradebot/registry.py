"""Model registry and champion/challenger promotion, shared by the ML and RL pipelines.

Every trained version is stored (params, metrics, model file in Supabase Storage). A version
becomes the champion only if it beats the current champion's holdout Sortino by a margin,
trades enough, and stays below the kill-switch drawdown on the holdout.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from tradebot.agents import BASELINE_FACTORIES, DEFAULT_CHAMPION, Agent
from tradebot.config import RiskParams, Settings
from tradebot.ml.walkforward import WalkForwardConfig, evaluate
from tradebot.store import Store

log = logging.getLogger(__name__)

BUCKET = "models"
PROMOTION_MARGIN = 0.05
MAX_HOLDOUT_DD = 0.20
MODEL_AGENTS = ("lgbm", "rl")


def versions(store: Store, agent_id: str) -> list[dict[str, Any]]:
    return store.select("model_versions", {"agent_id": agent_id})


def next_version(store: Store, agent_id: str) -> int:
    rows = versions(store, agent_id)
    return max((int(r["version"]) for r in rows), default=0) + 1


def champion_id(store: Store) -> str:
    rows = store.select("champion")
    return str(rows[0]["agent_id"]) if rows else DEFAULT_CHAMPION


def champion_version(store: Store, agent_id: str) -> dict[str, Any] | None:
    rows = [r for r in versions(store, agent_id) if r["status"] == "champion"]
    return rows[0] if rows else None


def live_version(store: Store, agent_id: str) -> dict[str, Any] | None:
    """The version an agent trades live: its champion version, else its latest version."""
    rows = versions(store, agent_id)
    if not rows:
        return None
    return champion_version(store, agent_id) or max(rows, key=lambda r: int(r["version"]))


def load_agent(store: Store, agent_id: str) -> Agent | None:
    row = live_version(store, agent_id)
    if row is None or not row.get("storage_path"):
        return None
    blob = store.download(BUCKET, row["storage_path"])
    version = int(row["version"])
    if agent_id == "lgbm":
        from tradebot.ml.model import ModelStrategy, load_model

        booster, cfg, extra = load_model(blob)
        return Agent("lgbm", "ml", ModelStrategy(booster, cfg, version), RiskParams(**extra["risk"]))
    if agent_id == "rl":
        from tradebot.rl.agent import RLStrategy

        strat = RLStrategy.from_bytes(blob, version)
        return Agent("rl", "rl", strat, strat.risk)
    raise ValueError(f"unknown model agent {agent_id}")


def load_model_agents(store: Store) -> list[Agent]:
    """All trained model agents for live trading. A broken model never stops trading."""
    out = []
    for agent_id in MODEL_AGENTS:
        try:
            agent = load_agent(store, agent_id)
        except Exception:
            log.exception("could not load agent %s", agent_id)
            continue
        if agent is not None:
            out.append(agent)
    return out


def evaluate_champion(
    settings: Settings, store: Store, dfs: dict[str, pd.DataFrame], wf: WalkForwardConfig,
    start: pd.Timestamp, lgbm_signals: dict[str, pd.Series] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Current champion's metrics on the holdout starting at ``start``.

    ``lgbm_signals`` lets the LightGBM pipeline pass walk-forward signals for the champion
    configuration (a stored final model was trained on recent data, so replaying it on the
    holdout would leak)."""
    cid = champion_id(store)
    if cid == "lgbm" and lgbm_signals is not None:
        row = champion_version(store, "lgbm")
        risk = RiskParams(**row["params"]["risk"]) if row else settings.risk
        return cid, evaluate(dfs, dict(lgbm_signals), risk, settings.costs, wf, start)
    if cid in MODEL_AGENTS:
        agent = load_agent(store, cid)
        if agent is not None:
            return cid, evaluate(dfs, {s: agent.strategy for s in dfs}, agent.risk or settings.risk,
                                 settings.costs, wf, start)
    cid = cid if cid in BASELINE_FACTORIES else DEFAULT_CHAMPION
    strat = BASELINE_FACTORIES[cid]()
    return cid, evaluate(dfs, {s: strat for s in dfs}, settings.risk, settings.costs, wf, start)


def rejection_reasons(challenger: dict[str, Any], champion: dict[str, Any], min_trades: int) -> list[str]:
    """Every promotion rule the challenger fails; empty means it may become the champion."""
    reasons = []
    if not challenger["score"] > champion["score"] + PROMOTION_MARGIN:
        reasons.append(f"sortino {challenger['score']:.2f} not above champion "
                       f"{champion['score']:.2f} + {PROMOTION_MARGIN}")
    if not challenger["num_trades"] >= min_trades:
        reasons.append(f"trades {challenger['num_trades']:.0f} below minimum {min_trades}")
    if not challenger["max_drawdown"] < MAX_HOLDOUT_DD:
        reasons.append(f"max drawdown {challenger['max_drawdown']:.1%} not below {MAX_HOLDOUT_DD:.0%}")
    return reasons


def is_better(challenger: dict[str, Any], champion: dict[str, Any], min_trades: int) -> bool:
    return not rejection_reasons(challenger, champion, min_trades)


def record_version(
    store: Store, settings: Settings, *, agent_id: str, kind: str, version: int, blob: bytes,
    params: dict[str, Any], challenger: dict[str, Any], champion: dict[str, Any], champ_id: str,
    holdout: tuple[pd.Timestamp, pd.Timestamp], train_range: tuple[Any, Any], min_trades: int,
) -> bool:
    """Store a trained version, its holdout metrics and backtest rows; promote if better."""
    reasons = rejection_reasons(challenger, champion, min_trades)
    promoted = not reasons
    outcome = "promoted" if promoted else f"not promoted: {'; '.join(reasons)}"
    log.info("%s v%d %s", agent_id, version, outcome)
    path = f"{agent_id}/v{version}.json"
    store.upload(BUCKET, path, blob)
    now = pd.Timestamp.now(tz="UTC")
    label = f"{agent_id}(v={version})"
    metrics = {k: challenger.get(k) for k in ("score", "num_trades", "max_drawdown", "total_return",
                                              "sharpe", "tune_score")}
    metrics["sortino"] = challenger["score"]
    metrics["champion_score"] = champion["score"]

    store.upsert("agents", [{"id": agent_id, "kind": kind, "label": label, "params": params}],
                 on_conflict="id")
    if promoted:  # one champion at a time, across all agents
        for aid in MODEL_AGENTS:
            old = champion_version(store, aid)
            if old is not None:
                store.upsert("model_versions", [{"agent_id": aid, "version": old["version"],
                                                 "status": "retired"}], on_conflict="agent_id,version")
    store.upsert("model_versions", [{
        "agent_id": agent_id, "version": version, "status": "champion" if promoted else "rejected",
        "params": params, "metrics": metrics, "train_start": train_range[0], "train_end": train_range[1],
        "storage_path": path, "promoted_at": now if promoted else None,
    }], on_conflict="agent_id,version")
    store.insert("backtest_results", [{
        "agent_id": agent_id, "label": label, "symbol": s, "timeframe": settings.markets.trade_timeframe,
        "period_start": holdout[0], "period_end": holdout[1], "metrics": m,
    } for s, m in challenger["per_symbol"].items()])
    store.insert("events", [{
        "ts": now, "type": "retrain", "agent_id": agent_id,
        "message": (f"{label}: holdout Sortino {challenger['score']:.2f} vs champion {champ_id} "
                    f"{champion['score']:.2f} -> {outcome}"),
        "details": {"challenger": metrics, "rejection_reasons": reasons,
                    "champion": {k: champion[k] for k in ("score", "num_trades", "max_drawdown")}},
    }])
    if promoted:
        row = next(r for r in versions(store, agent_id) if int(r["version"]) == version)
        store.upsert("champion", [{"id": 1, "agent_id": agent_id, "version_id": row.get("id"),
                                   "since": now}], on_conflict="id")
        store.insert("events", [{"ts": now, "type": "champion_change", "agent_id": agent_id,
                                 "message": f"New champion: {label} (was {champ_id})", "details": {}}])
    return promoted


def record_baselines(store: Store, settings: Settings, dfs: dict[str, pd.DataFrame],
                     wf: WalkForwardConfig, start: pd.Timestamp, end: pd.Timestamp) -> None:
    rows = []
    for agent_id, make in BASELINE_FACTORIES.items():
        strat = make()
        r = evaluate(dfs, {s: strat for s in dfs}, settings.risk, settings.costs, wf, start)
        rows += [{"agent_id": agent_id, "label": strat.label, "symbol": s,
                  "timeframe": settings.markets.trade_timeframe, "period_start": start,
                  "period_end": end, "metrics": m} for s, m in r["per_symbol"].items()]
    store.insert("backtest_results", rows)


def trainer_status(store: Store, mode: str, details: dict[str, Any], error: str | None = None) -> None:
    now = pd.Timestamp.now(tz="UTC")
    row: dict[str, Any] = {"component": "trainer", "mode": mode, "last_step_at": now,
                           "last_error": error, "details": details}
    if error is None:
        row["last_ok_at"] = now
    store.upsert("bot_status", [row], on_conflict="component")
