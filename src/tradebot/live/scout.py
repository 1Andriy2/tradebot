"""The 5-minute paper lane: every fixed strategy on the busiest coins, checked each minute.

It runs beside the 15-minute step and changes nothing there. It has no venue, so it cannot place an
order, and its agents have kind `baseline`, so none can become champion. Strategies still decide for
themselves: a bar with no signal changes nothing. A coin that leaves the list trades no more, and its
open position is closed by its next bar.

The job runs for hours, so state lives in memory and reaches the database only when a position
changes or once an hour. A restart loses nothing: the missed bars are replayed from the saved bar.
Hold decisions are not stored: at this frequency they would fill the free database within months.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from tradebot.agents import BASELINE_FACTORIES, Agent
from tradebot.config import Settings
from tradebot.live.runner import StepReport, Trader
from tradebot.paper import EPS, PaperBook
from tradebot.policy import PolicyState
from tradebot.store import Store

log = logging.getLogger(__name__)

PREFIX = "s5_"
TIMEFRAME = "5m"
BAR_SECONDS = 300
POLL_SECONDS = 60
CLOSE_DELAY = 5  # the exchange needs a moment to publish a closed candle
SLOTS = 10  # a book's size stays the same however many coins the list holds this week
SKIPPED = ("chandelier_trend_4h",)  # a 4-hour strategy is not a 5-minute one


def scout_agents() -> list[Agent]:
    return [Agent(f"{PREFIX}{agent_id}", "baseline", make())
            for agent_id, make in BASELINE_FACTORIES.items() if agent_id not in SKIPPED]


class ScoutTrader(Trader):
    def __init__(self, settings: Settings, store: Store, public_client: Any, coins: list[str]) -> None:
        lane = settings.model_copy(deep=True)
        lane.markets.symbols = list(coins)
        lane.markets.trade_timeframe = TIMEFRAME
        super().__init__(lane, store, public_client, None, scout_agents())
        self.coins = list(coins)
        self.slot_equity = lane.backtest.initial_equity / SLOTS
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        for agent in self.agents:
            for row in store.select("agent_state", {"agent_id": agent.id, "venue": "shadow"}):
                self.rows[(agent.id, row["symbol"])] = row
        self.snapshot_hour: pd.Timestamp | None = None

    def step(self) -> StepReport:
        held = {s for (_, s), row in self.rows.items() if abs(row["state"]["book"]["units"]) > EPS}
        self.symbols = [*self.coins, *sorted(held - set(self.coins))]
        return super().step()

    def _prepare_news_twins(self) -> None:
        self._targets.clear()
        self.roster = list(self.agents)

    def _champion_id(self) -> str:
        return ""

    def _keep_candles(self, symbol: str, df: pd.DataFrame) -> None:
        """Candles stay out of the database: nothing reads them, and they would be most of its rows."""

    def _signal(self, agent: Agent, symbol: str, df: pd.DataFrame) -> pd.Series:
        if symbol not in self.coins:
            return pd.Series(0.0, index=df.index)
        return super()._signal(agent, symbol, df)

    def _record(self, decisions: list[dict[str, Any]]) -> None:
        super()._record([d for d in decisions if d["action"] != "hold"])

    def _load(self, agent_id: str, venue: str, symbol: str) -> dict[str, Any] | None:
        return self.rows.get((agent_id, symbol))

    def _save(self, agent_id: str, venue: str, symbol: str, state: PolicyState,
              book: PaperBook, last_bar: pd.Timestamp) -> None:
        old = self.rows.get((agent_id, symbol))
        self.rows[(agent_id, symbol)] = {
            "symbol": symbol, "last_bar": last_bar,
            "state": {"policy": state.to_dict(), "book": book.to_dict()},
        }
        moved = old is None or abs(old["state"]["book"]["units"] - book.units) > EPS
        if moved or last_bar.minute == 0:
            super()._save(agent_id, venue, symbol, state, book, last_bar)

    def _snapshot_equity(self) -> None:
        """Once an hour, as profit over the start, so a coin leaving the list moves nothing."""
        hour = pd.Timestamp.now(tz="UTC").floor("h")
        if hour == self.snapshot_hour:
            return
        rows = []
        for agent in self.agents:
            total = self.settings.backtest.initial_equity
            for (agent_id, symbol), row in self.rows.items():
                if agent_id != agent.id:
                    continue
                book = PaperBook.from_dict(row["state"]["book"], self.settings.costs)
                if symbol not in self.last_price and abs(book.units) > EPS:
                    total = float("nan")  # no price this step: a guess would draw a false jump
                    break
                total += book.equity(self.last_price.get(symbol, 0.0)) - self.slot_equity
            if total == total:
                rows.append({"ts": hour, "agent_id": agent.id, "venue": "shadow", "equity": total})
        self.store.upsert("equity_snapshots", rows, on_conflict="agent_id,venue,ts")
        self.snapshot_hour = hour

    def _status(self, report: StepReport) -> None:
        now = pd.Timestamp.now(tz="UTC")
        row: dict[str, Any] = {
            "component": "scout", "mode": "shadow-only", "last_step_at": now,
            "details": {"coins": self.coins, "decisions": report.decisions},
            "last_error": "; ".join(report.errors) or None,
        }
        if not report.errors:
            row["last_ok_at"] = now
        self.store.upsert("bot_status", [row], on_conflict="component")


def run_scout(trader: ScoutTrader, minutes: float, clock: Callable[[], float] = time.time,
              sleep: Callable[[float], None] = time.sleep) -> None:
    """Look every minute and step once per closed bar. A failed step is tried again next minute."""
    deadline = clock() + minutes * 60
    done = None
    while (now := clock()) < deadline:
        bar = int(now // BAR_SECONDS)
        if bar != done and now % BAR_SECONDS >= CLOSE_DELAY:
            try:
                log.info("scout step: %s", trader.step())
                done = bar
            except Exception:
                log.exception("scout step failed")
        now = clock()
        sleep(POLL_SECONDS - now % POLL_SECONDS + CLOSE_DELAY)
