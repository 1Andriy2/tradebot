"""Perpetual funding-rate history with a local Parquet cache.

Funding is paid every 8h (00:00/08:00/16:00 UTC): a long pays ``rate * notional`` when the
rate is positive, a short receives it (and vice versa when negative).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EMPTY_STEP_MS = 30 * 24 * 3_600_000  # skip ahead when a window has no data (before listing)


def funding_path(data_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("/", "").replace(":", "_")
    return data_dir / f"{safe}_funding.parquet"


def fetch_funding_range(client: Any, symbol: str, since_ms: int, limit: int = 200) -> pd.Series:
    now = int(client.milliseconds())
    rows: dict[int, float] = {}
    cursor = since_ms
    while cursor < now:
        batch = client.fetch_funding_rate_history(symbol, since=cursor, limit=limit)
        if not batch:
            cursor += EMPTY_STEP_MS
            continue
        for r in batch:
            rows[int(r["timestamp"])] = float(r["fundingRate"])
        last = int(batch[-1]["timestamp"])
        if last < cursor:
            break
        cursor = last + 1
    s = pd.Series(rows, dtype=float).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    s.name = "funding_rate"
    return s


def update_funding(client: Any, data_dir: Path, symbol: str, start: str) -> pd.Series:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = funding_path(data_dir, symbol)
    cached = pd.read_parquet(path)["funding_rate"] if path.exists() else pd.Series(dtype=float)
    since = (int(cached.index[-1].timestamp() * 1000) + 1 if len(cached)
             else int(pd.Timestamp(start, tz="UTC").timestamp() * 1000))
    fresh = fetch_funding_range(client, symbol, since)
    merged = pd.concat([cached, fresh]) if len(cached) else fresh
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_frame("funding_rate").to_parquet(path)
    log.info("%s funding: %d rates (+%d)", symbol, len(merged), len(fresh))
    return merged


def load_funding(data_dir: Path, symbol: str) -> pd.Series | None:
    path = funding_path(data_dir, symbol)
    return pd.read_parquet(path)["funding_rate"] if path.exists() else None


def funding_per_bar(index: pd.DatetimeIndex, funding: pd.Series | None) -> pd.Series:
    """Funding rate charged at the open of each bar (0 where no funding event falls in it).

    A funding event at time t is assigned to the bar containing t, so positions held into
    that bar's open pay it — consistent with fills happening at bar opens."""
    out = np.zeros(len(index))
    if funding is not None and not funding.empty and len(index) >= 2:
        f = funding[(funding.index >= index[0]) & (funding.index <= index[-1])]
        pos = index.searchsorted(f.index, side="right") - 1
        np.add.at(out, pos[pos >= 0], f.to_numpy(float)[pos >= 0])
    return pd.Series(out, index=index)
