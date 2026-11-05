"""One trading step (cron mode) and the daemon loop.

Every step, for each symbol, on the last fully closed candle:
- every agent advances its shadow paper book (same rules as the backtester);
- the champion agent also trades the Bybit demo account (if a venue is configured);
- news twins hold each original's position scaled by the news filter, in shadow books only.
Each step is idempotent per bar, so a doubled cron run does not trade twice.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradebot import indicators as ind
from tradebot.agents import (
    DEFAULT_CHAMPION,
    NEWS_REFERENCES,
    NEWS_TWIN_KINDS,
    Agent,
    all_agents,
    news_twin,
)
from tradebot.config import Settings
from tradebot.data.history import fetch_ohlcv_range
from tradebot.exchange.venue import Venue
from tradebot.news.gate import exposure_scale, load_levels
from tradebot.paper import EPS, PaperBook
from tradebot.policy import DecisionPolicy, PolicyState
from tradebot.risk import RiskManager
from tradebot.simulate import Bar, step_bar
from tradebot.store import Store

log = logging.getLogger(__name__)

LOOKBACK_BARS = 1000  # warm-up for the slowest feature (1h EMA-192)
STOP_REASONS = ("stop_loss", "take_profit")


@dataclass(frozen=True)
class BarResult:
    """What an agent did on one bar, for its news twin to follow."""

    target: float
    signal: float
    reason: str
    stop_price: float | None = None  # filled inside the bar, at this price
    stop_reason: str = ""


@dataclass
class StepReport:
    bar_time: dict[str, str] = field(default_factory=dict)
    decisions: int = 0
    orders: int = 0
    errors: list[str] = field(default_factory=list)


def _twin_reason(step: BarResult) -> str:
    """Why a twin's position changed: the original's reason, or the filter if the original held."""
    return "news_filter" if step.reason == "no_change" else step.reason


def _action(before: float, after: float) -> str:
    if abs(after - before) < EPS:
        return "hold"
    if abs(before) < EPS:
        return "open"
    if abs(after) < EPS:
        return "close"
    if np.sign(before) != np.sign(after):
        return "flip"
    return "increase" if abs(after) > abs(before) else "reduce"


class Trader:
    def __init__(
        self, settings: Settings, store: Store, public_client: Any, venue: Venue | None = None,
        agents: list[Agent] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.public = public_client
        self.venue = venue
        self.agents = agents if agents is not None else all_agents() + _ml_agents(store)
        self.policy = DecisionPolicy(RiskManager(settings.risk))
        self.policies = {a.id: DecisionPolicy(RiskManager(a.risk)) if a.risk else self.policy
                         for a in self.agents}
        self.symbols = settings.markets.symbols
        self.tf = settings.markets.trade_timeframe
        self.slot_equity = settings.backtest.initial_equity / len(self.symbols)
        self.last_price: dict[str, float] = {}
        self._long_history: dict[tuple[str, int], pd.DataFrame] = {}
        self.roster = list(self.agents)
        self.news_levels: pd.DataFrame | None = None  # None: unknown this step, not "no news"
        self._targets: dict[tuple[str, str], dict[pd.Timestamp, BarResult]] = {}

    # ---------- public API ----------

    def step(self) -> StepReport:
        report = StepReport()
        self._long_history.clear()
        champion = self._champion_id()
        self._prepare_news_twins()
        self.store.upsert("agents", [
            {"id": a.id, "kind": a.kind, "label": a.label, "params": a.strategy.params}
            for a in self.roster
        ], on_conflict="id")
        for symbol in self.symbols:
            try:
                df = self._recent_candles(symbol)
            except Exception as exc:
                self._error(report, f"{symbol}: {exc!r}")
                continue
            report.bar_time[symbol] = df.index[-1].isoformat()
            self.last_price[symbol] = float(df["close"].iloc[-1])
            atr = ind.atr(df, self.settings.risk.atr_period)
            # Each agent is isolated: one failing agent (or a demo-venue error) never stops the rest.
            for agent in self.roster:
                try:
                    if agent.news_gated:  # after its original, whose position it follows
                        report.decisions += self._advance_twin(agent, symbol, df)
                        continue
                    signal = self._signal(agent, symbol, df)
                    report.decisions += self._advance_shadow(agent, symbol, df, signal, atr)
                    if self.venue is not None and agent.id == champion:
                        report.orders += self._trade_demo(agent, symbol, df, signal, atr)
                except Exception as exc:
                    self._error(report, f"{agent.id}/{symbol}: {exc!r}", agent.id)
        self._snapshot_equity()
        self._status(report)
        return report

    def _prepare_news_twins(self) -> None:
        """Add the twins and read the levels. News problems never stop a step."""
        self._targets.clear()
        self.roster = list(self.agents)
        for base in self.agents:
            if base.kind in NEWS_TWIN_KINDS or base.id in NEWS_REFERENCES:
                twin = news_twin(base)
                self.roster.append(twin)
                self.policies[twin.id] = self.policies[base.id]
        try:
            self.news_levels = load_levels(self.store)
        except Exception as exc:
            log.warning("no news levels this step: %s", type(exc).__name__)
            self.news_levels = None

    def _error(self, report: StepReport, msg: str, agent_id: str | None = None) -> None:
        log.exception("step failed: %s", msg)
        report.errors.append(msg)
        self._event("error", msg, {"traceback": traceback.format_exc()[-4000:]}, agent_id)

    # ---------- data ----------

    def _recent_candles(self, symbol: str) -> pd.DataFrame:
        step_ms = int(self.public.parse_timeframe(self.tf) * 1000)
        since = int(self.public.milliseconds()) - LOOKBACK_BARS * step_ms
        df = fetch_ohlcv_range(self.public, symbol, self.tf, since)
        if len(df) < 50:
            raise RuntimeError(f"too few candles for {symbol}: {len(df)}")
        tail = df.tail(8).reset_index()
        self.store.upsert("candles", [
            {"symbol": symbol, "timeframe": self.tf, "ts": r.ts, "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "volume": r.volume}
            for r in tail.itertuples()
        ], on_conflict="symbol,timeframe,ts")
        return df

    def _signal(self, agent: Agent, symbol: str, df: pd.DataFrame) -> pd.Series:
        """The agent's signal on the step's bars, from a longer history if it needs one."""
        need = agent.strategy.history_bars
        if need <= LOOKBACK_BARS:
            return agent.strategy.signal(df)
        key = (symbol, need)
        if key not in self._long_history:
            step_ms = int(self.public.parse_timeframe(self.tf) * 1000)
            since = int(self.public.milliseconds()) - need * step_ms
            self._long_history[key] = fetch_ohlcv_range(self.public, symbol, self.tf, since)
        return agent.strategy.signal(self._long_history[key]).reindex(df.index, method="ffill")

    # ---------- state ----------

    def _load(self, agent_id: str, venue: str, symbol: str) -> dict[str, Any] | None:
        rows = self.store.select("agent_state", {"agent_id": agent_id, "venue": venue, "symbol": symbol})
        return rows[0] if rows else None

    def _save(self, agent_id: str, venue: str, symbol: str, state: PolicyState,
              book: PaperBook, last_bar: pd.Timestamp) -> None:
        payload = {"policy": state.to_dict(), "book": book.to_dict()}
        self.store.upsert("agent_state", [{
            "agent_id": agent_id, "venue": venue, "symbol": symbol, "state": payload,
            "last_bar": last_bar, "updated_at": pd.Timestamp.now(tz="UTC"),
        }], on_conflict="agent_id,venue,symbol")
        self.store.upsert("positions", [{
            "agent_id": agent_id, "venue": venue, "symbol": symbol, "units": book.units,
            "avg_entry": book.avg_entry or None, "stop_loss": state.sl, "take_profit": state.tp,
            "updated_at": pd.Timestamp.now(tz="UTC"),
        }], on_conflict="agent_id,venue,symbol")

    def _restore(self, row: dict[str, Any] | None, equity: float,
                 policy: DecisionPolicy) -> tuple[PolicyState, PaperBook]:
        if row is None:
            return policy.new_state(equity), PaperBook(equity, self.settings.costs)
        st = row["state"]
        return PolicyState.from_dict(st["policy"]), PaperBook.from_dict(st["book"], self.settings.costs)

    # ---------- shadow (paper) agents ----------

    def _advance_shadow(self, agent: Agent, symbol: str, df: pd.DataFrame,
                        signal: pd.Series, atr: pd.Series) -> int:
        policy = self.policies[agent.id]
        row = self._load(agent.id, "shadow", symbol)
        state, book = self._restore(row, self.slot_equity, policy)
        last_bar = pd.Timestamp(row["last_bar"]) if row and row.get("last_bar") else df.index[-2]
        new = df.index[df.index > last_bar]
        if len(new) == 0:
            return 0
        decisions = []
        targets = self._targets.setdefault((agent.id, symbol), {})
        o, h, lo, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        sig_v, atr_v = signal.to_numpy(float), atr.to_numpy(float)
        for i in range(len(df) - len(new), len(df)):  # new bars are the tail of df
            ts = df.index[i]
            bar = Bar(ts, ts.date(), o[i], h[i], lo[i], c[i], float(sig_v[i]),
                      float(atr_v[i - 1]) if i > 0 else float("nan"))
            before = book.units
            decision = step_bar(book, state, policy, bar, agent.strategy.uses_stops)
            after = decision.target_units if decision else book.units
            stop = next((f for f in book.fills if f["time"] == ts and f["reason"] in STOP_REASONS), None)
            targets[ts] = BarResult(after, bar.signal, decision.reason if decision else "no_change",
                                    float(stop["price"]) if stop else None,  # type: ignore[arg-type]
                                    str(stop["reason"]) if stop else "")
            decisions.append(self._decision_row(ts, agent.id, "shadow", symbol, bar.signal, before,
                                                after, decision.reason if decision else "no_change"))
        self._flush(agent.id, "shadow", symbol, state, book)
        self.store.insert("decisions", decisions)
        self._save(agent.id, "shadow", symbol, state, book, new[-1])
        return len(decisions)

    # ---------- news twins ----------

    def _advance_twin(self, agent: Agent, symbol: str, df: pd.DataFrame) -> int:
        """Hold the original's target for each new bar, scaled by the news level.

        A scaled copy of the original's position can never exceed it or sit on the other side, and
        the original's rules apply through it. A stop the original hit inside a bar closes the twin at
        the same price, and other orders fill at the next open, as they do for the original.
        """
        targets = self._targets.get((agent.id.removeprefix("news_"), symbol), {})
        row = self._load(agent.id, "shadow", symbol)
        state, book = self._restore(row, self.slot_equity, self.policies[agent.id])
        last_bar = pd.Timestamp(row["last_bar"]) if row and row.get("last_bar") else df.index[-2]
        new = df.index[df.index > last_bar]
        if len(new) == 0:
            return 0
        wanted = pd.Series({ts: targets[ts].target for ts in new if ts in targets}, dtype=float)
        bar = pd.Timedelta(seconds=self.public.parse_timeframe(self.tf))
        scale = exposure_scale(wanted, self._twin_levels(symbol, state), symbol.split("/")[0], bar)
        decisions = []
        opens = df["open"].to_numpy(float)
        for i in range(len(df) - len(new), len(df)):  # new bars are the tail of df
            ts = df.index[i]
            pending = state.extra.pop("pending", None)
            if pending is not None:
                book.fill(ts, pending["target"], float(opens[i]), pending["reason"])
            before = book.units
            step = targets.get(ts)
            if step is None:
                decisions.append(self._decision_row(ts, agent.id, "shadow", symbol, 0.0, before, before,
                                                    "no_change"))
                continue
            if step.stop_price is not None:
                book.fill(ts, 0.0, step.stop_price, step.stop_reason, slippage=False)
            after, moved = step.target * scale[ts], False
            if abs(after - book.units) > EPS:
                moved = True
                state.extra["pending"] = {"target": after, "reason": _twin_reason(step)}
            decisions.append(self._decision_row(ts, agent.id, "shadow", symbol, step.signal * scale[ts] + 0.0,
                                                before, after, _twin_reason(step) if moved else "no_change"))
        self._flush(agent.id, "shadow", symbol, state, book)
        self.store.insert("decisions", decisions)
        self._save(agent.id, "shadow", symbol, state, book, new[-1])
        return len(decisions)

    def _twin_levels(self, symbol: str, state: PolicyState) -> pd.DataFrame:
        """Levels to use now. In an outage, the last ones this twin saw: a failed download must not
        undo a cut, and they expire by age like any other level."""
        scope = symbol.split("/")[0]
        if self.news_levels is None:
            saved = state.extra.get("levels")
            if not saved:
                return pd.DataFrame()
            return pd.DataFrame([saved]).assign(ts=lambda d: pd.to_datetime(d["ts"], utc=True))
        mine = self.news_levels[self.news_levels["scope"] == scope] if len(self.news_levels) else None
        if mine is not None and len(mine):
            latest = mine.sort_values("ts").tail(1)
            state.extra["levels"] = {"ts": pd.Timestamp(latest["ts"].iloc[0]).isoformat(), "scope": scope,
                                     "long_mult": float(latest["long_mult"].to_numpy(float)[0]),
                                     "short_mult": float(latest["short_mult"].to_numpy(float)[0])}
        return self.news_levels

    # ---------- champion on the demo account ----------

    def _trade_demo(self, agent: Agent, symbol: str, df: pd.DataFrame,
                    signal: pd.Series, atr: pd.Series) -> int:
        assert self.venue is not None
        ts = df.index[-1]
        row = self._load(agent.id, "demo", symbol)
        if row and row.get("last_bar") and pd.Timestamp(row["last_bar"]) >= ts:
            return 0  # this bar was already handled
        policy = self.policies[agent.id]
        total_equity = self.venue.equity()
        sizing_equity = total_equity / len(self.symbols)
        state, ledger = self._restore(row, sizing_equity, policy)
        close = float(df["close"].iloc[-1])
        units = self.venue.position_units(symbol)

        if abs(ledger.units) > EPS and abs(units) < EPS:
            # Closed on the exchange by the attached stop-loss / take-profit.
            reason, level = self._exchange_exit(ledger.units, state, close)
            ledger.fill(ts, 0.0, level, reason, slippage=False)
            policy.on_stopped(state)
        elif abs(ledger.units - units) > max(EPS, 0.01 * abs(units)):
            self._event("info", f"{symbol}: ledger resynced to exchange position",
                        {"ledger": ledger.units, "exchange": units}, agent.id)
            ledger.adopt(ts, units, close)

        decision = policy.decide(state, ts.date(), sizing_equity, float(signal.iloc[-1]), close, units)
        orders = 0
        if decision is not None:
            orders = self._execute_demo(agent, symbol, units, decision.target_units, decision.reason,
                                        ts, close, float(atr.iloc[-1]), state, ledger)
        after = self.venue.position_units(symbol) if orders else units
        self.store.insert("decisions", [self._decision_row(
            ts, agent.id, "demo", symbol, float(signal.iloc[-1]), units,
            decision.target_units if decision else units,
            decision.reason if decision else "no_change",
            {"equity": total_equity, "executed_units": after},
        )])
        self._flush(agent.id, "demo", symbol, state, ledger)
        self._save(agent.id, "demo", symbol, state, ledger, ts)
        return orders

    def _execute_demo(self, agent: Agent, symbol: str, units: float, target: float, reason: str,
                      ts: pd.Timestamp, close: float, atr_now: float,
                      state: PolicyState, ledger: PaperBook) -> int:
        assert self.venue is not None
        policy = self.policies[agent.id]
        self.venue.prepare(symbol, policy.risk.leverage)
        orders = 0
        flip_or_close = abs(units) > EPS and (abs(target) < EPS or np.sign(target) != np.sign(units))
        reduce = abs(units) > EPS and np.sign(target) == np.sign(units) and abs(target) < abs(units)
        if flip_or_close or reduce:
            qty = -units if flip_or_close else target - units
            fill = self.venue.market_order(symbol, qty, True, close)
            if fill:
                orders += 1
                ledger.fill(ts, ledger.units + fill.qty, fill.price, reason, slippage=False)
            units = units + (fill.qty if fill else 0.0)
            state.sl = state.tp = None
        remaining = target - units
        if abs(target) > EPS and abs(remaining) > EPS and np.sign(remaining) == np.sign(target):
            sl, tp = (policy.risk.stop_levels(np.sign(target), close, atr_now)
                      if agent.strategy.uses_stops else (None, None))
            fill = self.venue.market_order(symbol, remaining, False, close, sl, tp)
            if fill:
                orders += 1
                ledger.fill(ts, ledger.units + fill.qty, fill.price, reason, slippage=False)
                state.sl, state.tp = sl, tp
        return orders

    @staticmethod
    def _exchange_exit(units: float, state: PolicyState, close: float) -> tuple[str, float]:
        long = units > 0
        if state.sl is not None and (close <= state.sl if long else close >= state.sl):
            return "stop_loss", state.sl
        if state.tp is not None and (close >= state.tp if long else close <= state.tp):
            return "take_profit", state.tp
        if state.sl is not None and state.tp is not None:
            # Price came back since; pick the level closer to the current price.
            nearer_sl = abs(close - state.sl) <= abs(close - state.tp)
            return ("stop_loss", state.sl) if nearer_sl else ("take_profit", state.tp)
        return "exchange_close", close

    # ---------- persistence helpers ----------

    def _flush(self, agent_id: str, venue: str, symbol: str, state: PolicyState, book: PaperBook) -> None:
        _, trades = book.drain()
        self.store.insert("trades", [{
            "agent_id": agent_id, "venue": venue, "symbol": symbol, **t,
        } for t in trades])
        for ev in state.risk.events:
            self._event(str(ev["type"]), f"{agent_id}/{venue}/{symbol}: {ev['type']}",
                        {k: (str(v) if k == "day" else v) for k, v in ev.items()}, agent_id)
        state.risk.events.clear()

    def _decision_row(self, ts: pd.Timestamp, agent_id: str, venue: str, symbol: str, signal: float,
                      before: float, after: float, reason: str,
                      details: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"ts": ts, "agent_id": agent_id, "venue": venue, "symbol": symbol, "signal": signal,
                "units_before": before, "units_after": after, "action": _action(before, after),
                "reason": reason, "details": details or {}}

    def _event(self, type_: str, message: str, details: dict[str, Any] | None = None,
               agent_id: str | None = None) -> None:
        self.store.insert("events", [{"ts": pd.Timestamp.now(tz="UTC"), "type": type_,
                                      "agent_id": agent_id, "message": message,
                                      "details": details or {}}])

    def _champion_id(self) -> str:
        rows = self.store.select("champion")
        return str(rows[0]["agent_id"]) if rows else DEFAULT_CHAMPION

    def _snapshot_equity(self) -> None:
        now = pd.Timestamp.now(tz="UTC").floor("min")
        rows = []
        for agent in self.roster:
            total = 0.0
            for symbol in self.symbols:
                row = self._load(agent.id, "shadow", symbol)
                if row is None:
                    total += self.slot_equity
                    continue
                book = PaperBook.from_dict(row["state"]["book"], self.settings.costs)
                if symbol not in self.last_price and abs(book.units) > EPS:
                    total = float("nan")  # no price this step: a guess would draw a false jump
                    break
                total += book.equity(self.last_price.get(symbol, 0.0))
            if total == total:
                rows.append({"ts": now, "agent_id": agent.id, "venue": "shadow", "equity": total})
        if self.venue is not None:
            try:
                rows.append({"ts": now, "agent_id": self._champion_id(), "venue": "demo",
                             "equity": self.venue.equity()})
            except Exception as exc:  # a demo-venue error never stops the shadow agents
                log.warning("no demo equity this step: %s", type(exc).__name__)
        self.store.upsert("equity_snapshots", rows, on_conflict="agent_id,venue,ts")

    def _status(self, report: StepReport) -> None:
        now = pd.Timestamp.now(tz="UTC")
        row: dict[str, Any] = {
            "component": "trader", "mode": "demo" if self.venue else "shadow-only",
            "last_step_at": now, "details": {"bars": report.bar_time, "decisions": report.decisions,
                                             "orders": report.orders},
            "last_error": "; ".join(report.errors) or None,
        }
        if not report.errors:
            row["last_ok_at"] = now
        self.store.upsert("bot_status", [row], on_conflict="component")


def _ml_agents(store: Store) -> list[Agent]:
    """Trained model agents (LightGBM, RL) from the registry."""
    from tradebot.registry import load_model_agents

    return load_model_agents(store)


def seconds_until_next_bar(now_s: float, timeframe_s: int, delay_s: int = 15) -> float:
    """Sleep target: shortly after the next candle closes."""
    return timeframe_s - (now_s % timeframe_s) + delay_s


def run_daemon(trader: Trader, timeframe_s: int) -> None:
    while True:
        report = trader.step()
        log.info("step done: %s", report)
        time.sleep(seconds_until_next_bar(time.time(), timeframe_s))
