"""Train a PPO agent in the historical simulator and run champion/challenger on the holdout.

The policy only ever sees data before the holdout. Locally/CI this runs with a small budget;
the Kaggle notebook (notebooks/rl_kaggle.ipynb) runs the same code with a large budget on a
free CPU session and records the version through the same registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tradebot import registry
from tradebot.config import RiskParams, Settings
from tradebot.ml.features import build_features
from tradebot.ml.walkforward import WalkForwardConfig, evaluate, holdout_start
from tradebot.rl.agent import RLStrategy
from tradebot.rl.env import FeatureScaler, TradingEnv
from tradebot.store import Store

log = logging.getLogger(__name__)

AGENT_ID = "rl"
# The RL agent sizes with a fixed exposure (inside the hard caps) and manages its own exits.
RL_RISK = RiskParams(position_frac=0.25, leverage=1.0, stop_atr_mult=None, take_atr_mult=None)


@dataclass(frozen=True)
class RLConfig:
    timesteps: int = 300_000
    episode_len: int = 2000
    dd_penalty: float = 2.0
    n_envs: int = 4
    seed: int = 7
    device: str = "auto"


def train_policy(dfs: dict[str, pd.DataFrame], settings: Settings, cfg: RLConfig) -> RLStrategy:
    feats = {s: build_features(df) for s, df in dfs.items()}
    pooled = pd.concat([f.dropna() for f in feats.values()])
    scaler = FeatureScaler.fit(pooled)
    series = []
    for s, df in dfs.items():
        ok = feats[s].notna().all(axis=1).to_numpy()
        series.append((scaler.transform(feats[s][ok]), df["close"].to_numpy(float)[ok]))
    exposure = RL_RISK.position_frac * RL_RISK.leverage

    def make() -> TradingEnv:
        return TradingEnv(series, settings.costs, exposure, cfg.episode_len, cfg.dd_penalty)

    env = make_vec_env(make, n_envs=cfg.n_envs, seed=cfg.seed)
    model = PPO("MlpPolicy", env, seed=cfg.seed, device=cfg.device, verbose=0,
                n_steps=1024, batch_size=256, gamma=0.995, ent_coef=0.01)
    model.learn(total_timesteps=cfg.timesteps)
    return RLStrategy(model, scaler, RL_RISK)


def train_and_register(
    settings: Settings, store: Store, dfs: dict[str, pd.DataFrame],
    cfg: RLConfig | None = None, wf: WalkForwardConfig | None = None,
) -> tuple[int, bool, dict[str, float]]:
    cfg = cfg or RLConfig()
    wf = wf or WalkForwardConfig(initial_equity=settings.backtest.initial_equity)
    ho_start = holdout_start(dfs, wf)
    end = max(df.index[-1] for df in dfs.values())
    train = {s: df[df.index < ho_start] for s, df in dfs.items()}

    strat = train_policy(train, settings, cfg)
    challenger = evaluate(dfs, {s: strat for s in dfs}, RL_RISK, settings.costs, wf, ho_start)
    champ_id, champion = registry.evaluate_champion(settings, store, dfs, wf, ho_start)

    version = registry.next_version(store, AGENT_ID)
    strat.params["v"] = version
    params = {"rl": cfg.__dict__, "risk": RL_RISK.model_dump()}
    first = min(df.index[0] for df in train.values())
    promoted = registry.record_version(
        store, settings, agent_id=AGENT_ID, kind="rl", version=version,
        blob=strat.to_bytes({"rl": cfg.__dict__}), params=params, challenger=challenger,
        champion=champion, champ_id=champ_id, holdout=(ho_start, end), train_range=(first, ho_start),
        min_trades=wf.min_trades,
    )
    registry.trainer_status(store, "rl", {"agent": AGENT_ID, "version": version, "promoted": promoted})
    summary = {"score": challenger["score"], "champion_score": champion["score"],
               "num_trades": challenger["num_trades"], "max_drawdown": challenger["max_drawdown"]}
    return version, promoted, summary
