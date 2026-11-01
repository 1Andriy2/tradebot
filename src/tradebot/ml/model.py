"""LightGBM direction model and the strategy that trades its probabilities."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from tradebot.ml.features import build_features, make_label
from tradebot.strategies.base import Strategy

DEFAULT_LGBM: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 400,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 7,
    "num_threads": 0,
}


@dataclass
class ModelConfig:
    horizon: int = 32                # bars ahead (32 x 15m = 8h; best in walk-forward)
    num_boost_round: int = 300
    lgbm: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM))
    threshold: float = 0.04          # trade only if |p - 0.5| > threshold (learnable)
    smooth: int = 4                  # EMA span over probabilities to reduce churn

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(**d)


def training_frame(dfs: dict[str, pd.DataFrame], horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Pooled (symbols stacked) features and labels; rows with missing values dropped."""
    xs, ys = [], []
    for df in dfs.values():
        x = build_features(df)
        y = make_label(df, horizon)
        ok = x.notna().all(axis=1) & y.notna()
        xs.append(x[ok])
        ys.append(y[ok])
    return pd.concat(xs), pd.concat(ys)


def fit(x: pd.DataFrame, y: pd.Series, cfg: ModelConfig) -> lgb.Booster:
    data = lgb.Dataset(x, label=y, free_raw_data=True)
    return lgb.train(cfg.lgbm, data, num_boost_round=cfg.num_boost_round)


def probs_to_signal(p: pd.Series, threshold: float, smooth: int) -> pd.Series:
    if smooth > 1:
        p = p.ewm(span=smooth, adjust=False).mean()
    edge = p - 0.5
    sig = pd.Series(0.0, index=p.index)
    sig[edge > threshold] = 1.0
    sig[edge < -threshold] = -1.0
    return sig.where(p.notna(), 0.0)


def predict_proba(booster: lgb.Booster, df: pd.DataFrame) -> pd.Series:
    x = build_features(df)
    ok = x.notna().all(axis=1)
    vals = np.full(len(df), np.nan)
    mask = ok.to_numpy()
    if mask.any():
        vals[mask] = np.asarray(booster.predict(x[mask]), dtype=float)
    return pd.Series(vals, index=df.index)


class ModelStrategy(Strategy):
    """Trades a trained booster: long when P(up) is high, short when low."""

    name = "lgbm"

    def __init__(self, booster: lgb.Booster, cfg: ModelConfig, version: int = 0) -> None:
        super().__init__(v=version, thr=round(cfg.threshold, 4))
        self.booster = booster
        self.cfg = cfg

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return probs_to_signal(predict_proba(self.booster, df), self.cfg.threshold, self.cfg.smooth)


def dump_model(booster: lgb.Booster, cfg: ModelConfig, extra: dict[str, Any]) -> bytes:
    return json.dumps({"config": cfg.to_dict(), "extra": extra,
                       "booster": booster.model_to_string()}).encode()


def load_model(blob: bytes) -> tuple[lgb.Booster, ModelConfig, dict[str, Any]]:
    d = json.loads(io.BytesIO(blob).read())
    return (lgb.Booster(model_str=d["booster"]), ModelConfig.from_dict(d["config"]), d.get("extra", {}))
