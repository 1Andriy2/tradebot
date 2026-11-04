from __future__ import annotations

import numpy as np
import pandas as pd

from tests.conftest import make_ohlcv
from tests.test_ml import synthetic
from tradebot.cli import SCENARIO_HORIZONS, _scenario_events
from tradebot.strategies.analogs import AnalogForecast, build_index, scenario_outlook, state_features


def test_state_features_use_only_the_past() -> None:
    df = synthetic(1500, 31)
    pd.testing.assert_frame_equal(state_features(df).iloc[:1000], state_features(df.iloc[:1000]))


def test_a_bar_never_reads_its_own_future() -> None:
    """Inside the training data every usable neighbour's outcome must already be known."""
    df = synthetic(3000, 32)
    index = build_index(df, 16)
    assert index is not None
    states = state_features(df).dropna().iloc[600:2400]
    idx, chosen = index.neighbours(states, 50)
    known = index.known_at[idx]
    times = states.index.to_numpy("datetime64[ns]")[:, None]
    assert (known[chosen] <= np.broadcast_to(times, chosen.shape)[chosen]).all()
    assert chosen.sum(axis=1).max() == 50


def test_changing_the_future_cannot_change_a_signal() -> None:
    df = synthetic(3000, 33)
    strat = AnalogForecast(k=50, horizon=16)
    strat.fit(df.iloc[:2000])
    before = strat.signal(df).iloc[:2500]
    shocked = df.copy()
    shocked.iloc[2500:, shocked.columns.get_indexer(["open", "high", "low", "close"])] *= 3.0
    assert (strat.signal(shocked).iloc[:2500] == before).all()


def test_training_window_score_is_not_inflated_by_self_matches() -> None:
    """With pure noise, trades inside the training window must not look like foresight."""
    rng = np.random.default_rng(5)
    df = make_ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 6000))))
    strat = AnalogForecast(k=50, horizon=16)
    strat.fit(df)
    sig = strat.signal(df)
    forward = (df["close"].shift(-16) / df["close"] - 1)
    active = (sig != 0) & forward.notna()
    assert active.sum() > 100
    assert abs(float(np.mean(np.sign(sig[active]) == np.sign(forward[active]))) - 0.5) < 0.08


def test_finds_a_repeating_pattern_out_of_sample() -> None:
    """After a sharp fall the price rebounds: analogs must learn that and go long."""
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.002, 12000)
    for t in range(200, 11900, 60):
        r[t:t + 4] = -0.012
        r[t + 4:t + 20] = 0.004
    df = make_ohlcv(100 * np.exp(np.cumsum(r)))
    strat = AnalogForecast(k=50, horizon=16)
    strat.fit(df.iloc[:8000])
    sig = strat.signal(df).iloc[8000:]
    forward = (df["close"].shift(-16) / df["close"] - 1).iloc[8000:]
    long = (sig > 0) & forward.notna()
    assert long.sum() > 30 and float(forward[long].mean()) > 0.005


def test_without_training_data_it_stays_flat() -> None:
    df = synthetic(600, 34)
    assert (AnalogForecast().signal(df) == 0).all()
    strat = AnalogForecast()
    strat.fit(df.iloc[:40])
    assert strat.index is None and (strat.signal(df) == 0).all()


def test_scenario_outlook_is_consistent() -> None:
    df = synthetic(6000, 35)
    out = scenario_outlook(df, 16, k=100)
    assert out is not None and out["n"] == 100
    assert out["q10"] <= out["q50"] <= out["q90"]
    assert out["typical_worst"] <= 0 <= out["typical_best"]
    assert 0 <= out["p_up"] <= 1 and 0 <= out["base_p_up"] <= 1


def test_scenario_outlook_needs_history() -> None:
    assert scenario_outlook(synthetic(120, 36), 16, k=100) is None


def test_scenario_events_cover_each_symbol_and_horizon() -> None:
    df = synthetic(96 * 400, 37)
    rows = _scenario_events({"BTC/USDT:USDT": df}, pd.Timestamp("2026-01-01", tz="UTC"))
    assert [r["details"]["horizon"] for r in rows] == list(SCENARIO_HORIZONS)
    assert all(r["type"] == "scenario" and r["details"]["symbol"] == "BTC/USDT:USDT" for r in rows)
