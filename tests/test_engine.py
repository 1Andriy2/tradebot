from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from tradebot.backtest.engine import run_backtest
from tradebot.config import CostsConfig, RiskParams
from tradebot.risk import HardLimits
from tradebot.strategies.base import Strategy
from tradebot.strategies.baselines import BuyHold, baseline_strategies

NO_COSTS = CostsConfig(fee_rate=0, slippage_rate=0)
NO_STOPS = RiskParams(position_frac=0.25, leverage=2.0, stop_atr_mult=None, take_atr_mult=None)
LOOSE = HardLimits(daily_loss_limit=1.0, kill_switch_drawdown=1.0)


class AlwaysLong(Strategy):
    name = "always_long"

    def generate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index)


SMALL = RiskParams(position_frac=0.1, leverage=1.0, stop_atr_mult=None, take_atr_mult=None)


def test_buy_hold_tracks_price_without_costs() -> None:
    df = make_ohlcv(np.linspace(100, 200, 50))
    res = run_backtest(df, BuyHold(), NO_COSTS, SMALL, 1000.0, limits=LOOSE)
    # Entry at bar 1 open (=bar 0 close=100), exit at last close (200); notional 10% of equity.
    assert res.equity.iloc[-1] == pytest.approx(1000 * (1 + 0.1 * 1.0))
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["exit_reason"] == "end_of_test"


def test_short_profits_when_price_falls() -> None:
    df = make_ohlcv(np.linspace(200, 100, 50))
    sig = pd.Series(-1.0, index=df.index)
    res = run_backtest(df, BuyHold(), NO_COSTS, SMALL, 1000.0, limits=LOOSE, signal=sig)
    assert res.equity.iloc[-1] == pytest.approx(1000 * (1 + 0.1 * 0.5))
    assert res.trades.iloc[0]["side"] == "short"


def test_fees_and_slippage_are_charged() -> None:
    df = make_ohlcv(np.full(20, 100.0))
    costs = CostsConfig(fee_rate=0.001, slippage_rate=0.0005)
    res = run_backtest(df, BuyHold(), costs, NO_STOPS, 1000.0, limits=LOOSE)
    assert res.equity.iloc[-1] < 1000.0
    assert res.trades.iloc[0]["fees"] > 0


def test_signal_fills_next_bar_no_lookahead() -> None:
    close = np.array([100, 100, 100, 150, 150, 150], dtype=float)
    df = make_ohlcv(close)
    # Signal turns on at bar 3 close (after the jump): must not capture the jump.
    sig = pd.Series([0, 0, 0, 1, 1, 1], index=df.index, dtype=float)
    res = run_backtest(df, BuyHold(), NO_COSTS, NO_STOPS, 1000.0, limits=LOOSE, signal=sig)
    assert res.equity.iloc[-1] == pytest.approx(1000.0)
    assert res.fills.iloc[0]["time"] == df.index[4]


def test_future_data_does_not_change_past_signals(random_walk: pd.DataFrame) -> None:
    for strat in baseline_strategies():
        full = strat.signal(random_walk)
        part = strat.signal(random_walk.iloc[:1500])
        pd.testing.assert_series_equal(full.iloc[:1500], part, check_names=False)


def test_stop_loss_closes_position() -> None:
    close = np.concatenate([np.full(30, 100.0), [100, 80, 80, 80]])
    df = make_ohlcv(close, spread=0.001)
    params = RiskParams(position_frac=0.25, leverage=1.0, stop_atr_mult=2.0, take_atr_mult=None)
    res = run_backtest(df, AlwaysLong(), NO_COSTS, params, 1000.0, limits=LOOSE)
    assert res.trades.iloc[0]["exit_reason"] == "stop_loss"
    # After the stop the unchanged signal must not re-enter.
    assert len(res.trades) == 1


def test_kill_switch_flattens_and_blocks() -> None:
    close = np.concatenate([np.full(5, 100.0), np.linspace(100, 10, 30), np.linspace(10, 100, 30)])
    df = make_ohlcv(close)
    limits = HardLimits(daily_loss_limit=1.0, kill_switch_drawdown=0.10)
    res = run_backtest(df, BuyHold(), NO_COSTS, NO_STOPS, 1000.0, limits=limits)
    assert res.trades.iloc[0]["exit_reason"] == "kill_switch"
    assert len(res.trades) == 1
    assert any(e["type"] == "kill_switch" for e in res.risk_events)


def test_hard_caps_clip_learned_params() -> None:
    df = make_ohlcv(np.linspace(100, 200, 50))
    greedy = RiskParams(position_frac=5.0, leverage=50.0, stop_atr_mult=None, take_atr_mult=None)
    res = run_backtest(df, BuyHold(), NO_COSTS, greedy, 1000.0, limits=LOOSE)
    # Entry at the 50% notional cap; growth above 120% of the cap is trimmed back.
    assert res.exposure.max() <= 0.25 * 2.0 * 1.2 + 0.02
    assert (res.fills["reason"] == "risk_cap").any()


def test_all_baselines_run(random_walk: pd.DataFrame) -> None:
    for strat in baseline_strategies():
        res = run_backtest(random_walk, strat, CostsConfig(), RiskParams(), 10_000.0)
        assert len(res.equity) == len(random_walk)
        assert np.isfinite(res.equity).all()
