"""Property-based checks of the backtest engine's accounting and risk invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import make_ohlcv
from tradebot.backtest.engine import run_backtest
from tradebot.config import CostsConfig, RiskParams
from tradebot.risk import HardLimits
from tradebot.strategies.baselines import BuyHold

LOOSE = HardLimits(daily_loss_limit=1.0, kill_switch_drawdown=1.0)


@st.composite
def scenario(draw: st.DrawFn) -> tuple[pd.DataFrame, pd.Series, RiskParams, CostsConfig, pd.Series]:
    n = draw(st.integers(30, 200))
    seed = draw(st.integers(0, 10_000))
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, draw(st.floats(0.001, 0.03)), n)))
    df = make_ohlcv(close, spread=0.002)
    sig = pd.Series(rng.choice([-1.0, 0.0, 1.0], n, p=[0.3, 0.4, 0.3]), index=df.index)
    stop = draw(st.one_of(st.none(), st.floats(0.5, 5.0)))
    risk = RiskParams(position_frac=draw(st.floats(0.01, 1.0)), leverage=draw(st.floats(0.5, 10.0)),
                      stop_atr_mult=stop, take_atr_mult=draw(st.one_of(st.none(), st.floats(0.5, 8.0))))
    costs = CostsConfig(fee_rate=draw(st.floats(0, 0.002)), slippage_rate=draw(st.floats(0, 0.002)))
    funding = pd.Series(rng.normal(0, 0.0005, n // 8), index=df.index[::8][: n // 8])
    return df, sig, risk, costs, funding


@settings(max_examples=60, deadline=None)
@given(scenario())
def test_equity_equals_initial_plus_trade_pnl(case: tuple) -> None:  # type: ignore[type-arg]
    df, sig, risk, costs, funding = case
    res = run_backtest(df, BuyHold(), costs, risk, 1000.0, limits=LOOSE, signal=sig, funding=funding)
    pnl = float(res.trades["pnl"].sum()) if len(res.trades) else 0.0
    assert res.equity.iloc[-1] == pytest.approx(1000.0 + pnl, abs=1e-6)  # flat at the end


@settings(max_examples=60, deadline=None)
@given(scenario())
def test_exposure_never_far_above_hard_cap(case: tuple) -> None:  # type: ignore[type-arg]
    df, sig, risk, costs, funding = case
    res = run_backtest(df, BuyHold(), costs, risk, 1000.0, limits=LOOSE, signal=sig, funding=funding)
    cap = 0.25 * 2.0
    # Trimmed at 120% of the cap on a bar close; one more bar of price movement can add more.
    assert res.exposure.abs().max() <= cap * 1.2 * 1.5 + 1e-9

