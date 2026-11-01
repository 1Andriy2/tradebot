from __future__ import annotations

from datetime import date

import pytest

from tradebot.config import RiskParams
from tradebot.policy import RISK_CAP, DecisionPolicy
from tradebot.risk import RiskManager

DAY = date(2026, 11, 1)


def test_oversized_position_is_trimmed_even_without_signal_change() -> None:
    policy = DecisionPolicy(RiskManager(RiskParams()))
    state = policy.new_state(2_000.0)
    state.prev_sig = 1.0  # signal unchanged: the normal rules would hold
    d = policy.decide(state, DAY, 2_000.0, 1.0, 100.0, units=30.0)
    assert d is not None and d.reason == RISK_CAP
    assert d.target_units == pytest.approx(0.25 * 2.0 * 2_000.0 / 100.0)  # 10 units


def test_position_within_tolerance_is_kept() -> None:
    policy = DecisionPolicy(RiskManager(RiskParams()))
    state = policy.new_state(2_000.0)
    state.prev_sig = 1.0
    assert policy.decide(state, DAY, 2_000.0, 1.0, 100.0, units=11.0) is None  # 110% of cap
