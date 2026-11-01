from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.test_ml import WF, synthetic
from tradebot.config import CostsConfig, Settings
from tradebot.registry import load_agent
from tradebot.rl.agent import RLStrategy
from tradebot.rl.env import TradingEnv
from tradebot.rl.train import RLConfig, train_and_register
from tradebot.store import MemoryStore


def test_env_reward_includes_costs_and_drawdown_penalty() -> None:
    feats = np.zeros((50, 3), dtype=np.float32)
    close = np.full(50, 100.0)
    env = TradingEnv([(feats, close)], CostsConfig(fee_rate=0.001, slippage_rate=0.0),
                     exposure_frac=1.0, episode_len=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4,)
    _, r_open, *_ = env.step(2)   # open long on a flat market: pay the fee
    _, r_hold, *_ = env.step(2)   # hold: no cost, no move
    assert r_open < 0 and r_hold == pytest.approx(0.0)


def test_env_episode_terminates() -> None:
    env = TradingEnv([(np.zeros((40, 2), dtype=np.float32), np.linspace(100, 110, 40))],
                     CostsConfig(), episode_len=5)
    env.reset(seed=1)
    done = False
    steps = 0
    while not done:
        *_, done, _, _ = env.step(1)
        steps += 1
    assert steps == 5


def test_rl_train_register_and_reload() -> None:
    n = 96 * 300
    dfs = {"BTC/USDT:USDT": synthetic(n, 3), "ETH/USDT:USDT": synthetic(n, 4)}
    store = MemoryStore()
    cfg = RLConfig(timesteps=2048, n_envs=2, episode_len=500, device="cpu")
    version, promoted, summary = train_and_register(Settings(), store, dfs, cfg, WF)
    assert version == 1 and np.isfinite(summary["score"])
    agent = load_agent(store, "rl")
    assert agent is not None and isinstance(agent.strategy, RLStrategy)
    assert agent.risk is not None and agent.risk.position_frac <= 0.25
    sig = agent.strategy.signal(dfs["BTC/USDT:USDT"].iloc[-1200:])
    assert set(np.unique(sig)) <= {-1.0, 0.0, 1.0}


def test_rl_signal_has_no_lookahead() -> None:
    dfs = {"X": synthetic(96 * 40, 5)}
    store = MemoryStore()
    from tradebot.rl.train import train_policy

    strat = train_policy(dfs, Settings(), RLConfig(timesteps=1024, n_envs=1, episode_len=300, device="cpu"))
    df = dfs["X"]
    full, part = strat.signal(df), strat.signal(df.iloc[:3000])
    pd.testing.assert_series_equal(full.iloc[1000:2990], part.iloc[1000:2990])
    assert store is not None
