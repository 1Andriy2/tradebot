from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from tests.test_ml import synthetic
from tradebot.backtest.engine import run_backtest
from tradebot.config import CostsConfig, RiskParams
from tradebot.data.funding import funding_per_bar
from tradebot.research.lab import ResearchConfig, deflated_sharpe, run_research
from tradebot.research.patterns import bh_qvalues, scan_seasonality
from tradebot.strategies.baselines import BuyHold
from tradebot.strategies.library import FAMILIES, Seasonality, make

NO_COSTS = CostsConfig(fee_rate=0, slippage_rate=0)
FLAT_RISK = RiskParams(position_frac=0.25, leverage=1.0, stop_atr_mult=None, take_atr_mult=None)


def test_funding_assigned_to_bar_containing_event() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    f = pd.Series([0.001, 0.002], index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 09:00"], utc=True))
    out = funding_per_bar(pd.DatetimeIndex(idx), f)
    assert out.tolist() == [0.001, 0.0, 0.002, 0.0]


def test_long_pays_positive_funding() -> None:
    df = make_ohlcv(np.full(30, 100.0))
    rates = pd.Series(0.001, index=df.index[5:25:8])
    res = run_backtest(df, BuyHold(), NO_COSTS, FLAT_RISK, 1000.0, funding=rates)
    paid = float(res.trades["funding"].sum())
    assert paid == pytest.approx(len(rates) * 0.25 * 1000 * 0.001, rel=1e-6)
    assert res.equity.iloc[-1] == pytest.approx(1000.0 - paid)


def test_all_library_families_are_causal() -> None:
    df = synthetic(4000, 9)
    for family, (_, grid) in FAMILIES.items():
        strat = make(family, grid[0])
        strat.fit(df.iloc[:2000])
        full, part = strat.signal(df), strat.signal(df.iloc[:3000])
        pd.testing.assert_series_equal(full.iloc[:3000], part, check_names=False, obj=family)


def test_seasonality_finds_injected_hour_effect() -> None:
    n = 24 * 400
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.002, n) + np.where(idx.hour == 14, 0.003, 0.0)
    df = make_ohlcv(100 * np.exp(np.cumsum(r)), freq="1h")
    df.index = idx
    rows = scan_seasonality(df, "X")
    hit = [p for p in rows if p["kind"] == "hour" and p["bucket"] == 14][0]
    assert hit["stable"] and hit["q_value"] < 0.01 and hit["mean_return"] > 0
    assert sum(p["stable"] for p in rows if p["kind"] == "hour") <= 3  # few false positives

    s = Seasonality(t_min=3.0)
    s.fit(df.iloc[: n // 2])
    assert s.side_by_hour.get(13) == 1.0  # bar at 13:00 closes into the 14:00 move


def test_bh_qvalues_monotone_and_bounded() -> None:
    q = bh_qvalues(np.array([0.01, 0.04, 0.03, 0.5]))
    assert (q >= np.array([0.01, 0.04, 0.03, 0.5]) - 1e-12).all() and (q <= 1).all()


def test_deflated_sharpe_penalises_many_trials() -> None:
    few = deflated_sharpe(0.02, [0.0, 0.01, 0.02], 3, 5000, 0.0, 3.0)
    many = deflated_sharpe(0.02, list(np.linspace(-0.02, 0.02, 200)), 200, 5000, 0.0, 3.0)
    assert 0 <= many < few <= 1


def test_research_run_produces_oos_results() -> None:
    dfs = {"BTC/USDT:USDT": synthetic(96 * 330, 11)}
    cfg = ResearchConfig(timeframes=("1h", "4h"), families=("ma_cross", "donchian", "seasonality"),
                         first_test="2022-05-01", train_months=3, test_months=2, workers=1)
    run_id, results = run_research(dfs, {"BTC/USDT:USDT": None}, CostsConfig(), RiskParams(), cfg)
    assert run_id and len(results) == 6
    for r in results:
        m = r["metrics"]
        assert 0 <= m["dsr"] <= 1 and 0 <= m["fold_win_rate"] <= 1
        assert m["n_folds"] >= 3 and len(r["params"]["folds"]) >= 3
        assert np.isfinite(m["total_return"])


def test_daily_coins_join_the_run_as_daily_tasks() -> None:
    from tradebot.research.lab import to_timeframe

    dfs = {"BTC/USDT:USDT": synthetic(96 * 330, 11)}
    other = to_timeframe(synthetic(96 * 330, 12), "1d")
    cfg = ResearchConfig(timeframes=("1d",), families=("ma_cross",), first_test="2022-05-01",
                         train_months=3, test_months=2, workers=1, robustness=False)
    _, results = run_research(dfs, {"BTC/USDT:USDT": None}, CostsConfig(), RiskParams(), cfg,
                              {"SOL/USDT:USDT": other})
    found = {(r["symbol"], r["timeframe"]) for r in results}
    assert found == {("BTC/USDT:USDT", "1d"), ("SOL/USDT:USDT", "1d")}
