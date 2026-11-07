from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from tradebot.agents import FEED_FACTORIES, all_agents
from tradebot.config import Settings
from tradebot.data.feeds import Feed, FeedMissing
from tradebot.live.runner import Trader
from tradebot.news.risk import news_flow
from tradebot.store import MemoryStore
from tradebot.strategies.flows import (
    LiqFade,
    LiqMagnet,
    LiqSweep,
    NewsBalance,
    NewsEach,
    bursts,
    liquidation_map,
)
from tradebot.strategies.levels import RareLevel, visit_share


def bars(n: int = 200, freq: str = "15min", start: float = 100.0) -> pd.DataFrame:
    return make_ohlcv(np.full(n, start), freq=freq, spread=0.001)


# Levels price seldom reaches

def test_visit_share_counts_only_earlier_bars() -> None:
    df = make_ohlcv(np.r_[np.full(10, 100.0), [130.0]], freq="15min")
    share = visit_share(df, 10, 0.005)
    assert np.isnan(share.iloc[9])
    assert share.iloc[-1] == 0.0  # 130 was never visited by the 10 bars before


def test_a_rare_high_is_sold_and_a_rare_low_bought() -> None:
    close = np.r_[np.full(60, 100.0), [104.0], np.full(5, 100.0), [96.0]]
    df = make_ohlcv(close, freq="15min", spread=0.001)
    sig = RareLevel(window=50, rare=0.02, usual=0.10).signal(df)
    assert sig.iloc[60] == -1.0
    assert sig.iloc[61] == 0.0  # back where price trades often
    assert sig.iloc[-1] == 1.0


# News

def events(rows: list[tuple[float, str, str]], now: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame([{"impact": impact, "confidence": 1.0, "kind": "official", "cluster": f"c{i}",
                          "born": now - pd.Timedelta(age), "first_seen": now - pd.Timedelta(seen)}
                         for i, (impact, age, seen) in enumerate(rows)])


NOW = pd.Timestamp("2026-11-07 12:00", tz="UTC")


def test_an_old_bad_story_is_outweighed_by_a_new_good_one() -> None:
    net, fresh = news_flow(events([(-0.9, "24h", "24h"), (0.6, "10min", "10min")], NOW), NOW)
    assert net > 0
    assert fresh == pytest.approx(0.6, rel=0.05)  # ten minutes of fading


def test_the_balance_adds_up_instead_of_saturating() -> None:
    one, _ = news_flow(events([(-0.9, "0h", "2h")], NOW), NOW)
    three, fresh = news_flow(events([(-0.9, "0h", "2h")] * 3, NOW), NOW)
    assert three == pytest.approx(3 * one)
    assert fresh == 0.0  # nothing new this run


def levels(df: pd.DataFrame, rows: list[tuple[int, float, float]]) -> Feed:
    """Level rows written two minutes after the close of bar ``i``."""
    step = pd.Timedelta("15min")
    return Feed(news=pd.DataFrame([{"ts": df.index[i] + step + pd.Timedelta("2min"), "scope": "BTC",
                                    "net": net, "fresh": fresh} for i, net, fresh in rows]))


def test_a_strong_new_story_opens_a_trade_for_a_few_hours_from_the_next_bar() -> None:
    df = bars()
    strat = NewsEach(threshold=0.4, hold_bars=16)
    strat.feed = levels(df, [(100, -1.0, -0.7), (150, 0.0, 0.2)])
    sig = strat.signal(df)
    assert sig.iloc[100] == 0.0  # written after that bar closed
    assert (sig.iloc[101:117] == -1.0).all()
    assert sig.iloc[117] == 0.0
    assert (sig.iloc[140:] == 0.0).all()  # a weak story is not traded


def test_the_balance_is_traded_while_it_lasts_and_dropped_when_it_is_old() -> None:
    df = bars()
    strat = NewsBalance(enter=0.7, leave=0.25)
    strat.feed = levels(df, [(100, 0.8, 0.0), (104, 0.4, 0.0), (108, 0.1, 0.0), (120, -0.9, 0.0)])
    sig = strat.signal(df)
    assert sig.iloc[101] == 1.0 and sig.iloc[105] == 1.0  # keeps it above the exit line
    assert sig.iloc[109] == 0.0
    assert sig.iloc[121] == -1.0
    assert sig.iloc[121 + 4] == 0.0  # no level for an hour: not trusted any more


def test_no_feed_means_wait_not_trade() -> None:
    with pytest.raises(FeedMissing):
        NewsEach().signal(bars())


# Liquidations

def liq(df: pd.DataFrame, at: dict[int, tuple[float, float]]) -> pd.DataFrame:
    table = pd.DataFrame(0.0, index=df.index[-96:], columns=["long", "short"])
    for i, (long, short) in at.items():
        table.loc[df.index[i], ["long", "short"]] = [long, short]
    return table


def test_a_burst_is_a_large_share_of_the_day() -> None:
    df = bars()
    table = liq(df, {150: (300_000.0, 0.0), 160: (0.0, 20_000.0), 170: (0.0, 400_000.0)})
    side = bursts(df, table, 0.15, 25_000.0)
    assert side.iloc[150] == 1.0 and side.iloc[170] == -1.0
    assert side.iloc[160] == 0.0


def test_after_longs_are_wiped_out_the_fade_buys_for_two_hours() -> None:
    df = bars()
    strat = LiqFade(hold_bars=8)
    strat.feed = Feed(liquidations=liq(df, {150: (300_000.0, 0.0)}))
    sig = strat.signal(df)
    assert (sig.iloc[150:158] == 1.0).all() and sig.iloc[158] == 0.0


def oi_rising(df: pd.DataFrame) -> pd.Series:
    hours = pd.date_range(df.index[0].floor("h"), df.index[-1].floor("h"), freq="1h")
    return pd.Series(np.linspace(1e8, 2e8, len(hours)), index=hours)


def test_the_map_puts_longs_below_and_shorts_above_and_sees_nothing_early() -> None:
    df = bars(400)
    oi = oi_rising(df)
    waiting = liquidation_map(df, oi, band=0.10, sweep=0.10)
    assert (waiting.iloc[:7] == 0).all().all()  # the second hour's rise is known when it is over
    assert waiting["below"].iloc[7] > 0
    assert waiting["above"].iloc[-1] > 0 and waiting["below"].iloc[-1] > 0
    later = oi.copy()
    later.iloc[-1] = 1e12  # a change in the last hour cannot reach bars before it is over
    assert liquidation_map(df, later, 0.10, 0.10).iloc[:-4].equals(waiting.iloc[:-4])


def test_a_level_the_price_crossed_is_gone() -> None:
    close = np.r_[np.full(200, 100.0), np.full(200, 85.0)]  # 10x longs liquidate near 90.5
    df = make_ohlcv(close, freq="15min", spread=0.001)
    waiting = liquidation_map(df, oi_rising(df.iloc[:200]), band=0.12, sweep=0.12)
    assert waiting["below"].iloc[150] > 0
    assert waiting["longs_close"].iloc[-1] == 0.0


def test_the_sweep_follows_a_burst_when_more_wait_beyond_it() -> None:
    df = bars(400)
    feed = Feed(liquidations=liq(df, {390: (300_000.0, 0.0)}), open_interest=oi_rising(df))
    follow, fade = LiqSweep(sweep=0.12, more=0.3), LiqSweep(sweep=0.01, more=0.3)
    follow.feed = fade.feed = feed
    assert follow.signal(df).iloc[390] == -1.0  # more longs wait below: the fall goes on
    assert fade.signal(df).iloc[390] == 1.0


def test_the_magnet_leans_toward_the_larger_pool() -> None:
    df = bars(400)
    strat = LiqMagnet(band=0.03)
    strat.feed = Feed(open_interest=oi_rising(df))
    assert set(strat.signal(df).unique()) <= {-1.0, 0.0, 1.0}
    strat.feed = Feed(open_interest=pd.Series(dtype=float))
    assert (strat.signal(df) == 0).all()


# The live step

class Client:
    def __init__(self) -> None:
        self.now_ms = int(pd.Timestamp("2026-11-07", tz="UTC").timestamp() * 1000)

    def parse_timeframe(self, _: str) -> int:
        return 900

    def milliseconds(self) -> int:
        return self.now_ms

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> list[list[float]]:
        start = self.now_ms - 300 * 900_000
        return [[start + i * 900_000, 100.0, 100.1, 99.9, 100.0, 1.0] for i in range(300)
                if since <= start + i * 900_000][:limit]

    def load_markets(self) -> dict[str, Any]:
        raise RuntimeError("exchange down")


def test_flow_agents_wait_when_their_data_is_missing_and_the_rest_trade(monkeypatch: Any) -> None:
    def unreadable(_: Any) -> pd.DataFrame:
        raise RuntimeError("storage down")

    monkeypatch.setattr("tradebot.live.runner.load_levels", unreadable)
    store = MemoryStore()
    agents = [a for a in all_agents() if a.id in ("ma_cross", *FEED_FACTORIES)]
    bot = Trader(Settings(), store, Client(), None, agents)
    report = bot.step()
    assert report.errors == []
    traded = {r["agent_id"] for r in store.select("agent_state")}
    assert "ma_cross" in traded and not traded & set(FEED_FACTORIES)
