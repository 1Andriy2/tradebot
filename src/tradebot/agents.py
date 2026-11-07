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
from tradebot.strategies.flows import LiqFade, LiqMagnet, LiqSweep, NewsBalance, NewsEach
from tradebot.strategies.levels import RareLevel
from tradebot.strategies.regimes import ChandelierTrend, OnTimeframe, RegimeSwitch, VolScaledTrend
from tradebot.strategies.trend import DailyTrend


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
    "rare_level": RareLevel,
    "trend_1d_long": DailyTrend,
}
# Strategies that need live data beside the candles. They cannot be backtested on years of history,
# so they stay out of research and of the nightly baseline records, and trade on paper.
FEED_FACTORIES: dict[str, Callable[[], Strategy]] = {
    "newsflow_each": NewsEach,
    "newsflow_balance": NewsBalance,
    "liq_fade": LiqFade,
    "liq_sweep": LiqSweep,
    "liq_magnet": LiqMagnet,
}

DEFAULT_CHAMPION = "ma_cross"
# Agents that get a news twin. Models can become champion, and a twin that came and went with the
# title would be left holding a position. The rest are steady: a mostly flat agent shows the filter
# nothing, while buy_hold is always long and ma_cross goes short as well.
NEWS_REFERENCES = ("buy_hold", "ma_cross")
NEWS_TWIN_KINDS = ("ml", "rl")


def news_twin(agent: Agent) -> Agent:
    """The same agent behind the news filter: it takes the original's position, scaled. Its kind
    keeps it out of promotion, since it exists only to be compared with the original."""
    return replace(agent, id=f"news_{agent.id}", kind="baseline", news_gated=True)


def all_agents() -> list[Agent]:
    factories = {**BASELINE_FACTORIES, **FEED_FACTORIES}
    return [Agent(agent_id, "baseline", make()) for agent_id, make in factories.items()]
