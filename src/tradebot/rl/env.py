"""Gymnasium trading environment built on historical candles.

Action: 0 = short, 1 = flat, 2 = long (exposure = action_side * exposure_frac of equity).
Reward per bar: log equity growth after fees/slippage, minus a penalty for every new
increase of the drawdown from the running equity peak.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
from gymnasium import spaces

from tradebot.config import CostsConfig

OBS_CLIP = 5.0
FloatArray = npt.NDArray[np.float64]
Obs = npt.NDArray[np.float32]


class FeatureScaler:
    """Standardisation fitted on training data only; NaNs become 0 after scaling."""

    def __init__(self, columns: list[str], mean: FloatArray, std: FloatArray) -> None:
        self.columns, self.mean, self.std = columns, mean, std

    @classmethod
    def fit(cls, feats: pd.DataFrame) -> FeatureScaler:
        std = feats.std().replace(0, 1.0).fillna(1.0)
        return cls(list(feats.columns), feats.mean().fillna(0.0).to_numpy(), std.to_numpy())

    def transform(self, feats: pd.DataFrame) -> Obs:
        x = (feats[self.columns].to_numpy(float) - self.mean) / self.std
        out: Obs = np.clip(np.nan_to_num(x, nan=0.0), -OBS_CLIP, OBS_CLIP).astype(np.float32)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureScaler:
        return cls(d["columns"], np.asarray(d["mean"]), np.asarray(d["std"]))


class TradingEnv(gym.Env[Obs, int]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self, series: list[tuple[Obs, FloatArray]], costs: CostsConfig,
        exposure_frac: float = 0.25, episode_len: int = 2000, dd_penalty: float = 2.0,
    ) -> None:
        """``series``: per symbol (scaled features [n, k], close prices [n])."""
        super().__init__()
        self.series = [s for s in series if len(s[1]) > episode_len + 2]
        if not self.series:
            raise ValueError("not enough data for one episode")
        self.cost_rate = costs.fee_rate + costs.slippage_rate
        self.exposure_frac = exposure_frac
        self.episode_len = episode_len
        self.dd_penalty = dd_penalty
        k = self.series[0][0].shape[1]
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(k + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # type: ignore[assignment]

    def _obs(self) -> Obs:
        out: Obs = np.append(self.feats[self.t], np.float32(self.pos)).astype(np.float32)
        return out

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        self.feats, self.prices = self.series[int(self.np_random.integers(len(self.series)))]
        self.t = int(self.np_random.integers(0, len(self.prices) - self.episode_len - 1))
        self.end = self.t + self.episode_len
        self.pos = 0.0
        self.equity = self.peak = 1.0
        self.dd = 0.0
        return self._obs(), {}

    def step(self, action: int) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        new_pos = float(action) - 1.0
        cost = abs(new_pos - self.pos) * self.exposure_frac * self.cost_rate
        ret = new_pos * self.exposure_frac * (self.prices[self.t + 1] / self.prices[self.t] - 1)
        growth = max(1e-6, 1 + ret - cost)
        self.equity *= growth
        self.peak = max(self.peak, self.equity)
        dd = 1 - self.equity / self.peak
        reward = float(np.log(growth) - self.dd_penalty * max(0.0, dd - self.dd))
        self.dd, self.pos = dd, new_pos
        self.t += 1
        done = self.t >= self.end
        return self._obs(), reward, done, False, {"equity": self.equity}
