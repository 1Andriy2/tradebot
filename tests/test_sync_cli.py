from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import make_ohlcv
from tradebot.cli import build_parser
from tradebot.data.history import cache_path, load_cached
from tradebot.data.sync import pull_cache, push_cache
from tradebot.store import MemoryStore


def test_push_then_pull_restores_cache(tmp_path: Path) -> None:
    store = MemoryStore()
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    df = make_ohlcv([1.0, 2.0, 3.0])
    df.to_parquet(cache_path(src, "BTC/USDT:USDT", "15m"))
    assert push_cache(store, src, ["BTC/USDT:USDT"], ["15m", "1h"]) == 1
    assert pull_cache(store, dst, ["BTC/USDT:USDT"], ["15m", "1h"]) == 1
    pd.testing.assert_frame_equal(load_cached(dst, "BTC/USDT:USDT", "15m"), df, check_freq=False)


def test_cli_parses_all_commands() -> None:
    p = build_parser()
    assert p.parse_args(["step", "--shadow-only"]).shadow_only
    assert p.parse_args(["retrain", "--sync", "--no-download"]).sync
    args = p.parse_args(["rl-train", "--timesteps", "10", "--device", "cpu"])
    assert args.timesteps == 10 and args.device == "cpu"
    assert p.parse_args(["download"]).no_download is False
