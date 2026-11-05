from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_live import FakePublic, FakeVenue
from tradebot.agents import Agent, news_twin
from tradebot.config import Settings
from tradebot.live.runner import Trader
from tradebot.news.archive import BUCKET
from tradebot.news.gate import exposure_scale
from tradebot.news.pipeline import RISK_FILE
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


def apply_levels(raw: pd.Series, table: pd.DataFrame, scope: str, bar: pd.Timedelta) -> pd.Series:
    return raw * exposure_scale(raw, table, scope, bar)


def test_a_bar_sees_only_levels_written_before_it_closed() -> None:
    written = START + pd.Timedelta("29min") + pd.Timedelta("30s")  # within a minute of 10:30
    gated = apply_levels(signal(), levels(written), "BTC", BAR)
    assert gated[INDEX[1]] == 1.0 and gated[INDEX[2]] == 0.0  # 10:15 bar closes 10:30: too close


def test_longs_and_shorts_use_their_own_multiplier() -> None:
    table = levels(START, long_mult=0.5, short_mult=0.0)
    assert apply_levels(signal(0.8), table, "BTC", BAR).iloc[0] == pytest.approx(0.4)
    assert apply_levels(signal(-0.8), table, "BTC", BAR).iloc[0] == 0.0


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
    return pd.Timestamp(public.rows[-1][0], unit="ms", tz="UTC") + BAR - pd.Timedelta("5min")


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


def test_a_failed_download_keeps_a_cut_in_force() -> None:
    """Regression: an outage read as 'no news' and put the twin back at full size."""
    store, bot = run_with_levels(0.0)
    store.files.clear()

    def outage(bucket: str, path: str) -> bytes:
        raise RuntimeError("503")

    store.download = outage  # type: ignore[method-assign]
    bot.public.add_bar(100.0)
    assert bot.step().errors == []
    assert store.select("decisions", {"agent_id": "news_ma_cross"})[-1]["signal"] == 0.0


def test_a_cut_from_an_old_outage_expires() -> None:
    store, bot = run_with_levels(0.0)

    def outage(bucket: str, path: str) -> bytes:
        raise RuntimeError("503")

    store.download = outage  # type: ignore[method-assign]
    for _ in range(6):
        bot.public.add_bar(100.0)
    assert bot.step().errors == []
    assert store.select("decisions", {"agent_id": "news_ma_cross"})[-1]["signal"] == 1.0


def twin_books(multipliers: list[float], value: float = 1.0) -> tuple[MemoryStore, list[float], list[float]]:
    """Runs a few bars with these long multipliers. Returns the original's and the twin's targets."""
    store = MemoryStore()
    bot = trader(store, value=value)
    for mult in multipliers:
        store_levels(store, levels(last_time(bot.public) + BAR, mult, mult))
        bot.public.add_bar(100.0)
        assert bot.step().errors == []
    rows = {a: [d["units_after"] for d in store.select("decisions", {"agent_id": a})]
            for a in ("ma_cross", "news_ma_cross")}
    return store, rows["ma_cross"], rows["news_ma_cross"]


def test_the_twin_is_always_a_share_of_the_original() -> None:
    _, plain, twin = twin_books([1.0, 0.5, 0.0, 0.5, 1.0])
    size = max(plain)
    assert size > 0 and twin[-5:] == pytest.approx([size, size / 2, 0.0, size / 2, size])


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_the_twin_never_exceeds_or_opposes_the_original(value: float) -> None:
    _, plain, twin = twin_books([0.5, 1.0, 0.0, 0.3, 1.0], value)
    assert all(abs(t) <= abs(p) + 1e-12 and t * p >= 0 for p, t in zip(plain, twin, strict=True))


class Stops(Strategy):
    """Long, with the stop at work: the bar after entry falls through it."""

    name = "stops"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


def test_a_stop_closes_the_twin_at_the_same_price_on_the_same_bar() -> None:
    store = MemoryStore()
    settings = Settings()
    settings.markets.symbols = ["BTC/USDT:USDT"]
    bot = Trader(settings, store, FakePublic(), None, [Agent("ma_cross", "baseline", Stops())])
    bot.public.add_bar(100.0)
    bot.step()
    bot.public.add_bar(100.0)
    bot.step()
    bot.public.add_bar(50.0)
    assert bot.step().errors == []
    plain = store.select("trades", {"agent_id": "ma_cross"})
    twin = store.select("trades", {"agent_id": "news_ma_cross"})
    assert plain and twin and twin[-1]["exit_reason"] == plain[-1]["exit_reason"] == "stop_loss"
    assert twin[-1]["exit_price"] == plain[-1]["exit_price"]
    assert twin[-1]["exit_time"] == plain[-1]["exit_time"]


def test_policy_and_simulation_know_nothing_about_news() -> None:
    root = Path(__file__).parents[1] / "src/tradebot"
    for name in ("policy.py", "simulate.py"):
        text = (root / name).read_text()
        assert "scale" not in text and "news" not in text


def test_every_model_and_reference_agent_gets_a_twin_whoever_is_champion() -> None:
    store = MemoryStore()
    store.upsert("champion", [{"id": 1, "agent_id": "rl"}], on_conflict="id")
    settings = Settings()
    settings.markets.symbols = ["BTC/USDT:USDT"]
    agents = [Agent("lgbm", "ml", Always()), Agent("rl", "rl", Always()), Agent("grid", "baseline", Always())]
    bot = Trader(settings, store, FakePublic(), None, agents)
    bot.public.add_bar(100.0)
    bot.step()
    assert {a.id for a in bot.roster} == {"lgbm", "rl", "grid", "news_lgbm", "news_rl"}


def test_a_corrupt_levels_file_does_not_stop_the_step() -> None:
    store = MemoryStore()
    store.upload(BUCKET, RISK_FILE, b"not parquet")
    bot = trader(store)
    bot.public.add_bar(100.0)
    assert bot.step().errors == []
