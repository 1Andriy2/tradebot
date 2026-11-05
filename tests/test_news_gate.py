from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from tests.test_live import FakePublic, FakeVenue
from tradebot.agents import Agent, news_twin
from tradebot.config import Settings
from tradebot.live.runner import Trader
from tradebot.news.gate import apply_levels
from tradebot.news.pipeline import BUCKET, RISK_FILE
from tradebot.store import MemoryStore
from tradebot.strategies.base import Strategy

BAR = pd.Timedelta("15min")
START = pd.Timestamp("2026-11-05 10:00", tz="UTC")
INDEX = pd.date_range(START, periods=8, freq="15min")


class Always(Strategy):
    name = "always"

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = value

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index)


def levels(ts: pd.Timestamp, long_mult: float = 0.0, short_mult: float = 1.0,
           scope: str = "BTC") -> pd.DataFrame:
    return pd.DataFrame([{"ts": ts, "scope": scope, "long_mult": long_mult, "short_mult": short_mult}])


def signal(value: float = 1.0) -> pd.Series:
    return pd.Series(value, index=INDEX)


def test_a_bar_sees_only_levels_written_before_it_closed() -> None:
    written = START + pd.Timedelta("31min")  # after the 10:15 bar closes at 10:30
    gated = apply_levels(signal(), levels(written), "BTC", BAR)
    assert gated[INDEX[1]] == 1.0 and gated[INDEX[2]] == 0.0


def test_longs_and_shorts_use_their_own_multiplier() -> None:
    table = levels(START, long_mult=0.5, short_mult=0.0)
    assert apply_levels(signal(0.8), table, "BTC", BAR).iloc[0] == pytest.approx(0.4)
    assert apply_levels(signal(-0.8), table, "BTC", BAR).iloc[0] == 0.0


def test_a_cut_short_is_a_plain_zero() -> None:
    gated = apply_levels(signal(-1.0), levels(START, short_mult=0.0), "BTC", BAR)
    assert str(gated.iloc[0]) == "0.0"


def test_the_latest_level_wins() -> None:
    table = pd.concat([levels(START, 0.0), levels(START + BAR, 0.5)])
    assert apply_levels(signal(), table, "BTC", BAR).iloc[1] == 0.5


def test_a_level_more_than_an_hour_old_is_ignored() -> None:
    gated = apply_levels(signal(), levels(START - pd.Timedelta("2h")), "BTC", BAR)
    assert (gated == 1.0).all()


def test_another_market_or_no_levels_leaves_the_signal_alone() -> None:
    assert (apply_levels(signal(), levels(START, scope="ETH"), "BTC", BAR) == 1.0).all()
    assert (apply_levels(signal(), pd.DataFrame(), "BTC", BAR) == 1.0).all()


@pytest.mark.parametrize(("long_mult", "short_mult"), [(5.0, 5.0), (-3.0, -3.0), (np.nan, np.nan)])
def test_a_damaged_file_can_never_raise_or_flip_exposure(long_mult: float, short_mult: float) -> None:
    table = levels(START, long_mult, short_mult)
    for value in (0.8, -0.8):
        gated = apply_levels(signal(value), table, "BTC", BAR)
        assert (gated.abs() <= abs(value)).all() and (gated * value >= 0).all()


def test_twin_cannot_be_promoted() -> None:
    twin = news_twin(Agent("rl", "rl", Always()))
    assert (twin.id, twin.kind, twin.news_gated) == ("news_rl", "baseline", True)
    assert twin.label == "always + news filter"


# The trading step

def trader(store: MemoryStore, venue: FakeVenue | None = None, value: float = 1.0) -> Trader:
    settings = Settings()
    settings.markets.symbols = ["BTC/USDT:USDT"]
    return Trader(settings, store, FakePublic(), venue, [Agent("ma_cross", "baseline", Always(value))])


def store_levels(store: MemoryStore, table: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    table.to_parquet(buffer)
    store.upload(BUCKET, RISK_FILE, buffer.getvalue())


def last_time(public: FakePublic) -> pd.Timestamp:
    return pd.Timestamp(public.rows[-1][0], unit="ms", tz="UTC") + BAR


def run_with_levels(long_mult: float, venue: FakeVenue | None = None) -> tuple[MemoryStore, Trader]:
    store = MemoryStore()
    bot = trader(store, venue)
    store_levels(store, levels(last_time(bot.public), long_mult))
    bot.public.add_bar(100.0)
    store_levels(store, levels(last_time(bot.public), long_mult))
    report = bot.step()
    assert report.errors == []
    return store, bot


def test_the_twin_joins_the_step_and_is_filtered() -> None:
    store, _ = run_with_levels(0.0)
    agents = {r["id"]: r for r in store.tables["agents"]}
    assert agents["news_ma_cross"]["kind"] == "baseline"
    plain = store.select("decisions", {"agent_id": "ma_cross"})[-1]
    twin = store.select("decisions", {"agent_id": "news_ma_cross"})[-1]
    assert plain["signal"] == 1.0 and twin["signal"] == 0.0
    assert {r["agent_id"] for r in store.tables["equity_snapshots"]} >= {"ma_cross", "news_ma_cross"}


def test_the_demo_account_ignores_news_levels() -> None:
    with_news, without_news = FakeVenue(), FakeVenue()
    run_with_levels(0.0, with_news)
    bot = trader(MemoryStore(), without_news)
    bot.public.add_bar(100.0)
    bot.step()
    assert with_news.orders and with_news.orders == without_news.orders


def test_the_twin_never_trades_on_the_demo_account() -> None:
    store, _ = run_with_levels(0.5, FakeVenue())
    assert store.select("decisions", {"agent_id": "news_ma_cross", "venue": "demo"}) == []


def test_missing_levels_do_not_stop_the_step() -> None:
    store = MemoryStore()
    bot = trader(store)
    bot.public.add_bar(100.0)
    assert bot.step().errors == []
    assert store.select("decisions", {"agent_id": "news_ma_cross"})[-1]["signal"] == 1.0


def test_a_corrupt_levels_file_does_not_stop_the_step() -> None:
    store = MemoryStore()
    store.upload(BUCKET, RISK_FILE, b"not parquet")
    bot = trader(store)
    bot.public.add_bar(100.0)
    assert bot.step().errors == []


def test_both_the_champion_and_the_reference_agent_get_a_twin() -> None:
    store = MemoryStore()
    store.upsert("champion", [{"id": 1, "agent_id": "rl"}], on_conflict="id")
    settings = Settings()
    settings.markets.symbols = ["BTC/USDT:USDT"]
    agents = [Agent("ma_cross", "baseline", Always()), Agent("buy_hold", "baseline", Always()),
              Agent("rl", "rl", Always(0.0))]
    bot = Trader(settings, store, FakePublic(), None, agents)
    bot.public.add_bar(100.0)
    assert bot.step().errors == []
    assert {a.id for a in bot.roster} == {"ma_cross", "buy_hold", "rl", "news_ma_cross", "news_buy_hold",
                                          "news_rl"}


def test_a_champion_that_is_not_loaded_gets_no_twin() -> None:
    store = MemoryStore()
    store.upsert("champion", [{"id": 1, "agent_id": "gone"}], on_conflict="id")
    bot = trader(store)
    bot.public.add_bar(100.0)
    bot.step()
    assert [a.id for a in bot.roster] == ["ma_cross", "news_ma_cross"]
