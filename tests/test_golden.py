"""Regression guard: metrics of every strategy family on frozen synthetic data.

If a change alters trading logic, this test fails. When the change is intended, regenerate:
    UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_ml import synthetic
from tradebot.backtest.engine import run_backtest
from tradebot.config import CostsConfig, RiskParams
from tradebot.metrics import compute_metrics
from tradebot.research.lab import to_timeframe
from tradebot.strategies.library import FAMILIES, make

GOLDEN = Path(__file__).parent / "golden" / "families.json"
KEYS = ("total_return", "max_drawdown", "num_trades", "win_rate")


def current() -> dict[str, dict[str, float]]:
    df = to_timeframe(synthetic(96 * 120, 21), "1h")
    out = {}
    for family, (_, grid) in FAMILIES.items():
        strat = make(family, grid[0])
        strat.fit(df.iloc[: len(df) // 2])
        res = run_backtest(df, strat, CostsConfig(), RiskParams(), 10_000.0)
        m = compute_metrics(res.equity, res.trades, df["close"])
        out[family] = {k: round(m[k], 8) for k in KEYS}
    return out


def test_family_metrics_match_golden() -> None:
    got = current()
    if os.environ.get("UPDATE_GOLDEN") or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
        pytest.skip("golden file (re)generated")
    expected = json.loads(GOLDEN.read_text())
    assert set(got) == set(expected), "strategy families changed; regenerate the golden file"
    for family, metrics in expected.items():
        for k, v in metrics.items():
            assert got[family][k] == pytest.approx(v, rel=1e-6, abs=1e-9), f"{family}.{k}"
