from tradebot.agents import NO_CHAMPION
from tradebot.registry import MAX_HOLDOUT_DD, choose_champion, is_better, rejection_reasons
from tradebot.store import MemoryStore

GOOD = {"score": 0.8, "total_return": 0.02, "num_trades": 50.0, "max_drawdown": 0.10}
PASSING_CHAMPION = {"score": 0.5, "total_return": 0.01, "num_trades": 40.0, "max_drawdown": 0.05}
LOSING_CHAMPION = {"score": -3.0, "total_return": -0.03, "num_trades": 70.0, "max_drawdown": 0.04}


def test_passing_challenger_has_no_reasons() -> None:
    assert rejection_reasons(GOOD, PASSING_CHAMPION, 30) == []
    assert is_better(GOOD, PASSING_CHAMPION, 30)


def test_each_failed_rule_is_named() -> None:
    bad = {"score": 0.52, "total_return": 0.01, "num_trades": 12.0, "max_drawdown": MAX_HOLDOUT_DD}
    reasons = rejection_reasons(bad, PASSING_CHAMPION, 30)
    assert [r.split()[0] for r in reasons] == ["sortino", "trades", "max"]
    assert not is_better(bad, PASSING_CHAMPION, 30)


def test_beating_a_losing_champion_is_not_enough_a_challenger_must_make_money() -> None:
    less_bad = {**GOOD, "score": -1.0, "total_return": -0.01}
    assert rejection_reasons(less_bad, LOSING_CHAMPION, 30)[0].startswith("loses")
    assert rejection_reasons(GOOD, LOSING_CHAMPION, 30) == []  # no margin over a champion that fails


def test_nan_score_is_rejected() -> None:
    assert not is_better({**GOOD, "score": float("nan")}, PASSING_CHAMPION, 30)


def test_the_best_passing_strategy_takes_over_a_losing_champion() -> None:
    store = MemoryStore()
    results = {"slow": GOOD, "slower": {**GOOD, "score": 1.5}, "loser": LOSING_CHAMPION,
               "few": {**GOOD, "score": 3.0, "num_trades": 5.0}}
    assert choose_champion(store, results, "rl", LOSING_CHAMPION, 30) == "slower"
    assert store.select("champion")[0]["agent_id"] == "slower"
    assert store.select("events")[0]["type"] == "champion_change"


def test_a_losing_champion_steps_down_even_with_nobody_to_replace_it() -> None:
    store = MemoryStore()
    assert choose_champion(store, {"loser": LOSING_CHAMPION}, "rl", LOSING_CHAMPION, 30) == NO_CHAMPION
    assert store.select("champion")[0]["agent_id"] == NO_CHAMPION
    assert choose_champion(store, {"loser": LOSING_CHAMPION}, NO_CHAMPION, {**GOOD, "num_trades": 0.0}, 30) \
        == NO_CHAMPION
    assert len(store.select("events")) == 1  # nothing changes on the next night


def test_a_passing_champion_stays_unless_beaten_by_a_margin() -> None:
    store = MemoryStore()
    close = {**GOOD, "score": PASSING_CHAMPION["score"] + 0.01}
    assert choose_champion(store, {"close": close}, "rl", PASSING_CHAMPION, 30) == "rl"
    assert store.select("champion") == []
