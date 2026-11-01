"""RL policy wrapped as a Strategy, plus (de)serialisation of trained models."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from tradebot.config import RiskParams
from tradebot.ml.features import build_features
from tradebot.rl.env import FeatureScaler
from tradebot.strategies.base import Strategy


class RLStrategy(Strategy):
    """Runs the policy bar by bar; its own last action is part of the observation."""

    name = "rl"
    uses_stops = False  # the policy manages exits itself; hard risk limits still apply

    def __init__(self, model: PPO, scaler: FeatureScaler, risk: RiskParams, version: int = 0) -> None:
        super().__init__(v=version)
        self.model = model
        self.scaler = scaler
        self.risk = risk

    def generate(self, df: pd.DataFrame) -> pd.Series:
        feats = build_features(df)
        valid = feats.notna().all(axis=1).to_numpy()
        x = self.scaler.transform(feats)
        n = len(x)
        # The policy depends on (features, previous position): evaluate all three positions in
        # batch, then walk forward choosing the row that matches the previous action.
        actions = np.empty((3, n), dtype=np.int64)
        for j, pos in enumerate((-1.0, 0.0, 1.0)):
            obs = np.hstack([x, np.full((n, 1), pos, dtype=np.float32)])
            acts, _ = self.model.predict(obs, deterministic=True)
            actions[j] = np.asarray(acts).reshape(-1)
        out = np.zeros(n)
        prev = 0.0
        for t in range(n):
            if not valid[t]:
                prev = 0.0
                continue
            prev = float(actions[int(prev) + 1, t]) - 1.0
            out[t] = prev
        return pd.Series(out, index=df.index)

    def to_bytes(self, extra: dict[str, Any] | None = None) -> bytes:
        buf = io.BytesIO()
        self.model.save(buf)
        return json.dumps({
            "policy_zip": base64.b64encode(buf.getvalue()).decode(),
            "scaler": self.scaler.to_dict(), "risk": self.risk.model_dump(), "extra": extra or {},
        }).encode()

    @classmethod
    def from_bytes(cls, blob: bytes, version: int = 0) -> RLStrategy:
        d = json.loads(blob)
        model = PPO.load(io.BytesIO(base64.b64decode(d["policy_zip"])), device="cpu")
        return cls(model, FeatureScaler.from_dict(d["scaler"]), RiskParams(**d["risk"]), version)
