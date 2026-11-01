"""Historical OHLCV download with a local Parquet cache."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(data_dir: Path, symbol: str, timeframe: str) -> Path:
    safe = symbol.replace("/", "").replace(":", "_")
    return data_dir / f"{safe}_{timeframe}.parquet"


def rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Convert ccxt OHLCV rows to a UTC-indexed DataFrame."""
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df.astype(float)


def _fetch_with_retry(
    client: Any, symbol: str, timeframe: str, since: int, limit: int,
    retries: int = 5, sleep: Callable[[float], None] = time.sleep,
) -> list[list[float]]:
    for attempt in range(retries):
        try:
            return list(client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit))
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            if attempt == retries - 1:
                raise
            delay = 2.0**attempt
            log.warning("fetch_ohlcv failed (%s), retry in %.0fs", exc, delay)
            sleep(delay)
    return []


def fetch_ohlcv_range(
    client: Any, symbol: str, timeframe: str, since_ms: int, until_ms: int | None = None,
    limit: int = 1000, sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Page through ``fetch_ohlcv`` from ``since_ms``; only fully closed candles are kept."""
    step_ms = int(client.parse_timeframe(timeframe) * 1000)
    now_ms = int(client.milliseconds())
    end_ms = min(until_ms or now_ms, now_ms)
    rows: list[list[float]] = []
    cursor = since_ms
    while cursor < end_ms:
        batch = _fetch_with_retry(client, symbol, timeframe, cursor, limit, sleep=sleep)
        if not batch:
            break
        rows.extend(batch)
        last = int(batch[-1][0])
        if last + step_ms <= cursor:
            break
        cursor = last + step_ms
    df = rows_to_frame(rows) if rows else pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return df
    # Drop the candle that is still open and anything past the requested end.
    closed_before = pd.Timestamp(now_ms - step_ms, unit="ms", tz="UTC")
    df = df[df.index <= closed_before]
    if until_ms is not None:
        df = df[df.index < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return df


def update_cache(
    client: Any, data_dir: Path, symbol: str, timeframe: str, start: str
) -> pd.DataFrame:
    """Download missing candles after the cached ones and rewrite the Parquet file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(data_dir, symbol, timeframe)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    if cached.empty:
        since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    else:
        step_ms = int(client.parse_timeframe(timeframe) * 1000)
        since_ms = int(cached.index[-1].timestamp() * 1000) + step_ms
    fresh = fetch_ohlcv_range(client, symbol, timeframe, since_ms)
    merged = pd.concat([cached, fresh]) if not cached.empty else fresh
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_parquet(path)
    log.info("%s %s: %d candles (+%d)", symbol, timeframe, len(merged), len(fresh))
    return merged


def load_cached(data_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = cache_path(data_dir, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"no cached data at {path}; run `tradebot download` first")
    return pd.read_parquet(path)
