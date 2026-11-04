from tradebot.registry import MAX_HOLDOUT_DD, is_better, rejection_reasons

CHAMPION = {"score": -6.0}
GOOD = {"score": -3.0, "num_trades": 50.0, "max_drawdown": 0.10}


def test_passing_challenger_has_no_reasons() -> None:
    assert rejection_reasons(GOOD, CHAMPION, 30) == []
    assert is_better(GOOD, CHAMPION, 30)


def test_each_failed_rule_is_named() -> None:
    bad = {"score": -5.98, "num_trades": 12.0, "max_drawdown": MAX_HOLDOUT_DD}
    reasons = rejection_reasons(bad, CHAMPION, 30)
    assert [r.split()[0] for r in reasons] == ["sortino", "trades", "max"]
    assert not is_better(bad, CHAMPION, 30)


def test_nan_score_is_rejected() -> None:
    assert not is_better({**GOOD, "score": float("nan")}, CHAMPION, 30)
