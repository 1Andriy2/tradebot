"""Registry of tradeable agents (baselines now; ML/RL agents register here later)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tradebot.config import RiskParams
from tradebot.strategies.base import Strategy
from tradebot.strategies.baselines import (
    BollingerBreakout,
    BuyHold,
    Grid,
    MACrossover,
    RSIMeanReversion,
)


@dataclass(frozen=True)
class Agent:
    id: str
    kind: str  # baseline | ml | rl
    strategy: Strategy
    risk: RiskParams | None = None  # None: use the configured default risk params


BASELINE_FACTORIES: dict[str, Callable[[], Strategy]] = {
    "buy_hold": BuyHold,
    "ma_cross": MACrossover,
    "rsi_mr": RSIMeanReversion,
    "bb_breakout": BollingerBreakout,
    "grid": Grid,
}

DEFAULT_CHAMPION = "ma_cross"


def all_agents() -> list[Agent]:
    return [Agent(agent_id, "baseline", make()) for agent_id, make in BASELINE_FACTORIES.items()]
