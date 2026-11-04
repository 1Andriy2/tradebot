from __future__ import annotations

import numpy as np

from tests.conftest import make_ohlcv
from tests.test_ml import synthetic
from tradebot import registry
from tradebot.agents import DEFAULT_CHAMPION, all_agents
from tradebot.live.runner import LOOKBACK_BARS
from tradebot.strategies.regimes import (
    ChandelierTrend,
    RegimeSwitch,
    VolScaledTrend,
    chandelier_positions,
    regime_signal,
    vol_rank,
    vol_scale,
)

NEW_AGENTS = ("regime_switch", "vol_scaled_trend", "chandelier_trend")


def test_regime_signal_quadrants() -> None:
    trending = np.array([True, True, False, False])
    hot = np.array([False, True, False, True])
    out = regime_signal(trending, hot, np.full(4, -1.0), np.full(4, 1.0))
    assert out.tolist() == [-1.0, -0.5, 1.0, 0.0]


def test_vol_scale_steps_down() -> None:
    assert vol_scale(np.array([0.1, 0.75, 0.8, 0.95, 0.99]), 0.75).tolist() == [1.0, 1.0, 0.5, 0.5, 0.0]


def test_vol_rank_rises_when_volatility_jumps() -> None:
    rng = np.random.default_rng(0)
    r = np.r_[rng.normal(0, 0.001, 600), rng.normal(0, 0.02, 60)]
    df = make_ohlcv(100 * np.exp(np.cumsum(r)))
    rank = vol_rank(df, 200)
    assert rank.iloc[:199].isna().all()
    assert rank.iloc[500:590].median() < 0.9 and rank.iloc[600:620].max() == 1


def run_chandelier(close: list[float], direction: list[float], max_hold: int = 100) -> list[float]:
    c = np.array(close)
    return chandelier_positions(np.array(direction), c, np.maximum.accumulate(c), np.minimum.accumulate(c),
                                np.full(len(c), 1.0), 2.0, max_hold).tolist()


def test_chandelier_trails_and_waits_for_the_next_flip() -> None:
    close = [10.0, 10, 11, 12, 13, 10.5, 12, 13]
    direction = [0.0, 1, 1, 1, 1, 1, 1, 1]
    assert run_chandelier(close, direction) == [0, 1, 1, 1, 1, 0, 0, 0]


def test_chandelier_short_mirrors_long() -> None:
    close = [20.0, 20, 19, 18, 17, 19.5, 18]
    direction = [0.0, -1, -1, -1, -1, -1, -1]
    assert run_chandelier(close, direction) == [0, -1, -1, -1, -1, 0, 0]


def test_chandelier_flip_reverses_at_once() -> None:
    assert run_chandelier([10.0, 10, 10, 10], [0.0, 1, -1, -1]) == [0, 1, -1, -1]


def test_time_stop_only_closes_trades_not_in_profit() -> None:
    flat = [10.0] * 7
    assert run_chandelier(flat, [0.0, 1, 1, 1, 1, 1, 1], max_hold=2) == [0, 1, 1, 1, 0, 0, 0]
    rising = [10.0, 10, 10.5, 10.6, 10.7, 10.8, 10.9]
    assert run_chandelier(rising, [0.0, 1, 1, 1, 1, 1, 1], max_hold=2) == [0, 1, 1, 1, 1, 1, 1]


def test_chandelier_manages_its_own_exit() -> None:
    assert ChandelierTrend.uses_stops is False
    assert RegimeSwitch.uses_stops is True


def test_new_strategies_trade_both_sides_and_respect_bounds() -> None:
    df = synthetic(6000, 21)
    for strat in (RegimeSwitch(), VolScaledTrend(), ChandelierTrend()):
        sig = strat.signal(df)
        assert sig.between(-1, 1).all()
        assert (sig > 0).any() and (sig < 0).any(), strat.name


def test_live_window_matches_full_history() -> None:
    """Live trading only sees the last LOOKBACK_BARS, so the signal must not need more."""
    df = synthetic(9000, 22)
    ends = range(2000, 9000, 25)
    for strat in (RegimeSwitch(), VolScaledTrend(), ChandelierTrend()):
        full = strat.signal(df).to_numpy()
        live = [strat.signal(df.iloc[end - LOOKBACK_BARS:end]).iloc[-1] for end in ends]
        agreement = np.mean(np.array(live) == full[[end - 1 for end in ends]])
        assert agreement >= 0.95, (strat.name, agreement)
        assert np.abs(live).sum() > 0, strat.name


def test_new_agents_are_shadow_only() -> None:
    agents = {a.id: a for a in all_agents()}
    assert set(NEW_AGENTS) <= set(agents)
    assert all(agents[a].kind == "baseline" for a in NEW_AGENTS)
    assert DEFAULT_CHAMPION not in NEW_AGENTS
    assert not set(NEW_AGENTS) & set(registry.MODEL_AGENTS)  # only model agents can be promoted
