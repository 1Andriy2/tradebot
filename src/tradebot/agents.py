"""Registry of tradeable agents (baselines now; ML/RL agents register here later)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from tradebot.config import RiskParams
from tradebot.strategies.base import Strategy
from tradebot.strategies.baselines import (
    BollingerBreakout,
    BuyHold,
    Grid,
    MACrossover,
    RSIMeanReversion,
)
from tradebot.strategies.regimes import ChandelierTrend, OnTimeframe, RegimeSwitch, VolScaledTrend


@dataclass(frozen=True)
class Agent:
    id: str
    kind: str  # baseline | ml | rl
    strategy: Strategy
    risk: RiskParams | None = None  # None: use the configured default risk params
    news_gated: bool = False  # True: signal is scaled by the news risk level (shadow book only)

    @property
    def label(self) -> str:
        return f"{self.strategy.label} + news filter" if self.news_gated else self.strategy.label


CHANDELIER_4H_HISTORY = 6400  # 400 four-hour candles of 15m bars: live signals match full history

BASELINE_FACTORIES: dict[str, Callable[[], Strategy]] = {
    "buy_hold": BuyHold,
    "ma_cross": MACrossover,
    "rsi_mr": RSIMeanReversion,
    "bb_breakout": BollingerBreakout,
    "grid": Grid,
    "regime_switch": RegimeSwitch,
    "vol_scaled_trend": VolScaledTrend,
    "chandelier_trend": ChandelierTrend,
    "chandelier_trend_4h": lambda: OnTimeframe(ChandelierTrend(), "4h", CHANDELIER_4H_HISTORY),
}

DEFAULT_CHAMPION = "ma_cross"


def news_twin(agent: Agent) -> Agent:
    """The same agent behind the news filter. Its kind keeps it out of promotion: only ``lgbm`` and
    ``rl`` can become champion, and this one exists only to be compared with the original."""
    return replace(agent, id=f"news_{agent.id}", kind="baseline", news_gated=True)


def all_agents() -> list[Agent]:
    return [Agent(agent_id, "baseline", make()) for agent_id, make in BASELINE_FACTORIES.items()]
