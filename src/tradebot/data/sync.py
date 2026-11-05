"""Mirror the local Parquet caches (candles, funding, market stats, news) to Supabase Storage.

Lets CI and Kaggle start from the full history (and keep training even if the exchange's
market-data API is unreachable from that network for a while).
"""

from __future__ import annotations

import logging
from pathlib import Path

from tradebot.data.funding import funding_path
from tradebot.data.history import cache_path
from tradebot.data.market_stats import liquidations_path, stats_path
from tradebot.data.news import news_path
from tradebot.store import Store

log = logging.getLogger(__name__)

BUCKET = "data"


def _paths(data_dir: Path, symbols: list[str], timeframes: list[str]) -> list[Path]:
    out = [cache_path(data_dir, s, tf) for s in symbols for tf in timeframes]
    out += [funding_path(data_dir, s) for s in symbols]
    out += [stats_path(data_dir, s) for s in symbols]
    out += [liquidations_path(data_dir, s) for s in symbols]
    return [*out, news_path(data_dir)]


def pull_cache(store: Store, data_dir: Path, symbols: list[str], timeframes: list[str]) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in _paths(data_dir, symbols, timeframes):
        try:
            path.write_bytes(store.download(BUCKET, path.name))
            n += 1
        except Exception as exc:  # missing object on first run
            log.warning("no remote cache for %s: %s", path.name, exc)
    return n


def push_cache(store: Store, data_dir: Path, symbols: list[str], timeframes: list[str]) -> int:
    n = 0
    for path in _paths(data_dir, symbols, timeframes):
        if path.exists():
            store.upload(BUCKET, path.name, path.read_bytes())
            n += 1
    return n
