"""Mirror the local Parquet candle cache to Supabase Storage.

Lets CI and Kaggle start from the full history (and keep training even if the exchange's
market-data API is unreachable from that network for a while).
"""

from __future__ import annotations

import logging
from pathlib import Path

from tradebot.data.history import cache_path
from tradebot.store import Store

log = logging.getLogger(__name__)

BUCKET = "data"


def pull_cache(store: Store, data_dir: Path, symbols: list[str], timeframes: list[str]) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for symbol in symbols:
        for tf in timeframes:
            path = cache_path(data_dir, symbol, tf)
            try:
                path.write_bytes(store.download(BUCKET, path.name))
                n += 1
            except Exception as exc:  # missing object on first run
                log.warning("no remote cache for %s: %s", path.name, exc)
    return n


def push_cache(store: Store, data_dir: Path, symbols: list[str], timeframes: list[str]) -> int:
    n = 0
    for symbol in symbols:
        for tf in timeframes:
            path: Path = cache_path(data_dir, symbol, tf)
            if path.exists():
                store.upload(BUCKET, path.name, path.read_bytes())
                n += 1
    return n
