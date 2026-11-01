from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.config import Settings
from tradebot.ml.features import build_features, make_label
from tradebot.ml.model import ModelConfig, dump_model, fit, load_model, training_frame
from tradebot.ml.pipeline import AGENT_ID, retrain
from tradebot.ml.walkforward import WalkForwardConfig, walk_forward_probs
from tradebot.registry import load_agent
from tradebot.store import MemoryStore


def synthetic(n: int, seed: int) -> pd.DataFrame:
    """15m candles with a weak, learnable momentum component."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    eps = rng.normal(0, 0.003, n)
    for i in range(1, n):
        r[i] = 0.15 * r[i - 1] + eps[i]
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.001, n))
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread), "close": close,
        "volume": rng.lognormal(3, 0.3, n),
    }, index=idx)


FAST = ModelConfig(num_boost_round=20)
WF = WalkForwardConfig(first_test="2022-07-01", train_months=4, test_months=2,
                       holdout_months=2, n_trials=3, min_trades=1)


@pytest.fixture(scope="module")
def dfs() -> dict[str, pd.DataFrame]:
    n = 96 * 300  # ~10 months
    return {"BTC/USDT:USDT": synthetic(n, 1), "ETH/USDT:USDT": synthetic(n, 2)}


def test_features_have_no_lookahead(dfs: dict[str, pd.DataFrame]) -> None:
    df = dfs["BTC/USDT:USDT"].iloc[:5000]
    full = build_features(df)
    cut = 4000
    part = build_features(df.iloc[:cut])
    # Rows that are complete in both (the last partial hour may legitimately differ).
    pd.testing.assert_frame_equal(full.iloc[1000:cut - 4], part.iloc[1000:cut - 4])


def test_label_uses_future_close_only_for_target(dfs: dict[str, pd.DataFrame]) -> None:
    df = dfs["BTC/USDT:USDT"].iloc[:100]
    y = make_label(df, 4)
    assert y.iloc[-4:].isna().all()
    assert set(y.dropna().unique()) <= {0.0, 1.0}


def test_model_roundtrip(dfs: dict[str, pd.DataFrame]) -> None:
    x, y = training_frame({k: v.iloc[:6000] for k, v in dfs.items()}, FAST.horizon)
    booster = fit(x, y, FAST)
    b2, cfg, extra = load_model(dump_model(booster, FAST, {"risk": {"position_frac": 0.1}}))
    assert cfg == FAST and extra["risk"]["position_frac"] == 0.1
    np.testing.assert_allclose(booster.predict(x.iloc[:50]), b2.predict(x.iloc[:50]))


def test_walk_forward_predicts_only_after_first_test(dfs: dict[str, pd.DataFrame]) -> None:
    probs = walk_forward_probs(dfs, FAST, WF)
    p = probs["BTC/USDT:USDT"]
    assert p[p.index < pd.Timestamp(WF.first_test, tz="UTC")].isna().all()
    assert p[p.index >= pd.Timestamp(WF.first_test, tz="UTC")].notna().mean() > 0.9


def test_retrain_stores_version_and_live_agent_loads(dfs: dict[str, pd.DataFrame]) -> None:
    settings = Settings()
    store = MemoryStore()
    res = retrain(settings, store, dfs, WF, FAST)
    versions = store.select("model_versions", {"agent_id": AGENT_ID})
    assert len(versions) == 1 and versions[0]["version"] == 1
    assert versions[0]["status"] == ("champion" if res.promoted else "rejected")
    assert any(e["type"] == "retrain" for e in store.select("events"))
    assert store.select("backtest_results")
    agent = load_agent(store, AGENT_ID)
    assert agent is not None and agent.risk is not None
    assert agent.risk.position_frac <= 0.25 and agent.risk.leverage <= 2.0
    sig = agent.strategy.signal(dfs["BTC/USDT:USDT"].iloc[-1500:])
    assert sig.between(-1, 1).all()

    # Second night: version 2; at most one champion at any time.
    retrain(settings, store, dfs, WF, FAST)
    statuses = [v["status"] for v in store.select("model_versions", {"agent_id": AGENT_ID})]
    assert len(statuses) == 2 and statuses.count("champion") <= 1
