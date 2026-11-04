from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_ohlcv
from tests.test_ml import synthetic
from tradebot.config import CostsConfig, RiskParams
from tradebot.research import robustness as rb
from tradebot.research.lab import ResearchConfig, run_research
from tradebot.store import _clean
from tradebot.strategies.library import FAMILIES, make


def hourly(values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2022-01-01", periods=len(values), freq="1h", tz="UTC"))


def test_daily_returns_compound_to_total() -> None:
    equity = hourly(100 * np.exp(np.cumsum(np.full(24 * 30, 0.001))))
    daily = rb.daily_returns(equity)
    assert np.prod(1 + daily) == pytest.approx(equity.iloc[-1] / equity.resample("1D").last().iloc[0])


def test_tail_metrics_on_known_values() -> None:
    assert rb.ulcer_index(pd.Series([100.0, 110.0, 120.0])) == 0
    assert rb.ulcer_index(pd.Series([100.0, 50.0])) == pytest.approx(math.sqrt(0.125))
    r = pd.Series(np.arange(-10, 10) / 100)
    assert rb.cvar(r, 0.1) == pytest.approx(-0.095)
    assert rb.tail_ratio(pd.Series([0.0, 0.0])) is None


def test_tail_ratio_sees_fat_right_tail() -> None:
    r = pd.Series(np.r_[np.full(50, -0.01), np.full(50, 0.03)])
    ratio = rb.tail_ratio(r)
    assert ratio == pytest.approx(3.0)


def test_stationary_bootstrap_block_length() -> None:
    idx = rb.stationary_bootstrap_indices(1000, 200, 10, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 1000
    continued = np.mean(np.diff(idx, axis=1) % 1000 == 1)
    assert continued == pytest.approx(0.9, abs=0.01)


def test_rule_significance_false_positive_rate_on_noise() -> None:
    """On a signal unrelated to returns, p < 0.05 should show up about 5% of the time."""
    hits = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        close = hourly(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 24 * 300))))
        blocks = np.repeat(rng.choice([-1.0, 0.0, 1.0], 24 * 300 // 12), 12)
        p = rb.rule_significance(pd.Series(blocks, index=close.index), close, 300, 10, rng)
        assert p is not None
        hits += p < 0.05
    assert hits <= 14


def test_rule_significance_detects_real_timing() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.005, 24 * 300)
    close = hourly(100 * np.exp(np.cumsum(r)))
    log_next = np.log(close).diff().shift(-1).fillna(0.0)
    noisy = np.sign(log_next + rng.normal(0, 0.02, len(r)))  # weak knowledge of the next bar
    assert rb.rule_significance(noisy, close, 1000, 10, rng) < 0.01


def test_rule_significance_does_not_reward_drift() -> None:
    rng = np.random.default_rng(2)
    close = hourly(100 * np.exp(np.cumsum(rng.normal(0.001, 0.003, 24 * 300))))
    always_long = pd.Series(1.0, index=close.index)
    assert rb.rule_significance(always_long, close, 1000, 10, rng) > 0.2


def test_rule_significance_needs_enough_days() -> None:
    close = hourly(np.full(24 * 10, 100.0))
    rng = np.random.default_rng(0)
    assert rb.rule_significance(close * 0, close, 100, 10, rng) is None


def test_monte_carlo_flags_ruin_only_when_it_can_happen() -> None:
    rng = np.random.default_rng(3)
    idx = rb.stationary_bootstrap_indices(200, 500, 10, rng)
    safe = rb.monte_carlo(np.full(200, 0.001), idx, 0.2)
    assert safe["mc_loss_prob"] == 0 and safe["mc_ruin_prob"] == 0 and safe["mc_drawdown_p95"] == 0
    bad = rb.monte_carlo(rng.normal(-0.003, 0.02, 200), idx, 0.2)
    assert bad["mc_loss_prob"] > 0.5 and bad["mc_ruin_prob"] > 0.3
    assert bad["mc_return_p05"] < bad["mc_return_p50"]


def test_monte_carlo_counts_drop_from_starting_equity() -> None:
    idx = np.zeros((1, 3), dtype=np.int64)
    assert rb.monte_carlo(np.array([-0.5, 0.0, 0.0]), idx, 0.2)["mc_ruin_prob"] == 1


def test_cpcv_splits_purge_around_test_folds() -> None:
    n, purge = 120, 3
    splits = rb.cpcv_splits(n, 6, 2, purge)
    folds = np.array_split(np.arange(n), 6)
    assert len(splits) == 15
    for train, combo in splits:
        for f in combo:
            lo, hi = folds[f][0] - purge, folds[f][-1] + purge
            assert not ((train >= lo) & (train <= hi)).any()
    assert len(splits[0][0]) == 120 - 40 - purge  # folds 0 and 1 touch the start


def test_cpcv_paths_each_cover_the_full_history() -> None:
    m = np.random.default_rng(0).normal(0, 0.01, (300, 4))
    paths = rb.cpcv_paths(m)
    assert len(paths) == 5
    assert all(len(p) == 300 for p in paths)


def test_cpcv_selects_the_config_with_the_edge() -> None:
    rng = np.random.default_rng(4)
    m = rng.normal(0, 0.01, (800, 5))
    m[:, 2] += 0.004
    stats = rb.cpcv_stats(m)
    assert stats["cpcv_positive_paths"] == 1 and stats["cpcv_sharpe_median"] > 2


def test_pbo_near_half_on_noise_and_low_with_a_real_edge() -> None:
    values = [rb.pbo(np.random.default_rng(s).normal(0, 0.01, (400, 20))) for s in range(12)]
    assert 0.35 < float(np.mean(values)) < 0.65
    rng = np.random.default_rng(99)
    m = rng.normal(0, 0.01, (400, 20))
    m[:, 7] += 0.005
    assert rb.pbo(m) < 0.05


def test_pbo_undefined_cases() -> None:
    assert rb.pbo(np.zeros((400, 1))) is None
    assert rb.pbo(np.zeros((10, 5))) is None


def test_perturbed_params_stay_valid_and_constructible() -> None:
    df = synthetic(600, 5)
    for family, (_, grid) in FAMILIES.items():
        for params in grid:
            for moved in rb.perturbed_params(params):
                assert moved != params
                changed = [k for k in params if moved[k] != params[k]]
                assert len(changed) == 1
                strat = make(family, moved)
                strat.fit(df)
                assert strat.signal(df).between(-1, 1).all(), (family, moved)


def test_perturbed_params_drop_crossed_averages_and_keep_ints() -> None:
    moved = rb.perturbed_params({"fast": 50, "slow": 100})
    assert all(p["fast"] < p["slow"] for p in moved)
    assert all(isinstance(v, int) for p in moved for v in p.values())
    assert rb.perturbed_params({}) == []


def test_param_stability() -> None:
    assert rb.param_stability(2.0, [2.0, 1.0, 0.9, -1.0]) == 0.5
    assert rb.param_stability(-1.0, [1.0]) is None
    assert rb.param_stability(1.0, []) is None


def test_verdict_needs_profit_and_every_available_check() -> None:
    good = {"total_return": 0.5, "dsr": 0.99, "signal_q": 0.01, "pbo": 0.1, "cpcv_positive_paths": 1.0,
            "param_stability": None}
    assert rb.verdict(good) == {"checks_passed": 5 - 1, "checks_total": 4, "robust": 1.0}
    assert rb.verdict({**good, "total_return": -0.1})["robust"] == 0
    assert rb.verdict({**good, "pbo": 0.6})["robust"] == 0
    assert rb.verdict({"total_return": 1.0, "dsr": 0.99})["robust"] == 0  # too few checks


def research(symbols: tuple[str, ...]) -> list[dict[str, Any]]:
    dfs = {s: synthetic(96 * 330, 11 + i) for i, s in enumerate(symbols)}
    cfg = ResearchConfig(timeframes=("1h", "4h"), families=("ma_cross", "seasonality", "buy_hold"),
                         first_test="2022-05-01", train_months=3, test_months=2, workers=1, mc_sims=100)
    return run_research(dfs, dict.fromkeys(symbols), CostsConfig(), RiskParams(), cfg)[1]


def test_research_results_carry_robustness_and_stay_serialisable() -> None:
    results = research(("BTC/USDT:USDT", "ETH/USDT:USDT"))
    for r in results:
        m = r["metrics"]
        assert "_returns" not in r
        _clean(r)
        for key in ("ulcer_index", "cvar_5", "signal_p", "signal_q", "checks_total"):
            assert m[key] is not None, (r["family"], key)
        assert ("mc_loss_prob" in m) == (m["num_trades"] > 0)
        assert 0 <= m["signal_q"] <= 1 and m["checks_passed"] <= m["checks_total"]
        assert m["peer_pair_profitable"] is not None and m["peer_tf_profitable"] is not None
        assert m["pbo_universe"] is not None
    assert all(r["metrics"]["pbo"] is None for r in results if r["family"] == "buy_hold")
    assert any(r["metrics"]["pbo"] is not None for r in results if r["family"] == "ma_cross")


def test_single_symbol_has_no_pair_peers() -> None:
    assert all(r["metrics"]["peer_pair_profitable"] is None for r in research(("BTC/USDT:USDT",)))


def test_robustness_can_be_switched_off() -> None:
    dfs = {"X": make_ohlcv(100 + np.sin(np.arange(96 * 200) / 50), freq="15min")}
    cfg = ResearchConfig(timeframes=("1h",), families=("buy_hold",), first_test="2024-03-01",
                         train_months=1, test_months=1, workers=1, robustness=False)
    (result,) = run_research(dfs, {"X": None}, CostsConfig(), RiskParams(), cfg)[1]
    assert "checks_total" not in result["metrics"] and "_returns" not in result
