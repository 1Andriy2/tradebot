"""Open interest, long/short ratio and real liquidations, collected into Parquet caches.

Exchanges keep only a short history of these, so the value is in collecting them from now on.
Only OKX is supported: its public API publishes every liquidation order with price and side.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import ccxt
import httpx
import pandas as pd

log = logging.getLogger(__name__)

LIQUIDATIONS_URL = "https://www.okx.com/api/v5/public/liquidation-orders"
BACKFILL_DAYS = 30
PAGE = 100
MAX_LIQUIDATION_PAGES = 300  # about 30 000 orders per run: a stuck cursor cannot loop forever
HOUR_MS = 3_600_000
RETRIES = 3
RETRY_DELAY = 2.0
LIQUIDATION_KEY = ["ts", "side", "price", "size"]


def _safe(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "_")


def stats_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{_safe(symbol)}_stats.parquet"


def liquidations_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / f"{_safe(symbol)}_liquidations.parquet"


def _attempt(fetch: Any, symbol: str, cursor: int) -> list[dict[str, Any]] | None:
    """One page, retried on network errors. None means the exchange did not answer."""
    for attempt in range(RETRIES):
        try:
            batch: list[dict[str, Any]] = fetch(symbol, "1h", since=cursor, limit=PAGE)
            return batch
        except ccxt.NetworkError as exc:
            log.warning("%s attempt %d failed: %s", symbol, attempt + 1, exc)
            time.sleep(RETRY_DELAY)
    return None


def _paged(fetch: Any, symbol: str, since_ms: int, now_ms: int) -> list[dict[str, Any]]:
    """Walk a ccxt history method forward, keeping what arrived if the exchange stops answering."""
    rows: dict[int, dict[str, Any]] = {}
    cursor = since_ms
    while cursor < now_ms:
        batch = _attempt(fetch, symbol, cursor)
        fresh = [r for r in batch or [] if int(r["timestamp"]) not in rows]
        if not fresh:
            break
        rows.update({int(r["timestamp"]): r for r in fresh})
        cursor = max(rows) + HOUR_MS
    return [rows[k] for k in sorted(rows)]


def _since(cached: pd.DataFrame, column: str, now_ms: int) -> int:
    """Per column, so a gap in one series is filled on the next run."""
    known = cached[column].dropna() if column in cached else pd.Series(dtype=float)
    if known.empty:
        return now_ms - BACKFILL_DAYS * 24 * HOUR_MS
    return int(known.index[-1].timestamp() * 1000) + 1


def update_stats(client: Any, data_dir: Path, symbol: str) -> pd.DataFrame:
    """Hourly open interest and long/short ratio, merged into the cache."""
    path = stats_path(data_dir, symbol)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    now = int(client.milliseconds())
    oi = {int(r["timestamp"]): float(r["openInterestValue"]) for r in _paged(
        client.fetch_open_interest_history, symbol, _since(cached, "open_interest", now), now)}
    ratio = {int(r["timestamp"]): float(r["longShortRatio"]) for r in _paged(
        client.fetch_long_short_ratio_history, symbol, _since(cached, "long_short_ratio", now), now)}
    fresh = pd.DataFrame({"open_interest": pd.Series(oi, dtype=float),
                          "long_short_ratio": pd.Series(ratio, dtype=float)})
    fresh.index = pd.to_datetime(fresh.index, unit="ms", utc=True)
    merged = fresh.combine_first(cached) if len(cached) else fresh
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.sort_index().to_parquet(path)
    log.info("%s stats: %d hours (+%d)", symbol, len(merged), len(fresh))
    return merged


def parse_liquidations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows from an OKX liquidation-orders response. ``side`` is the position that was closed."""
    return [{"ts": int(d["ts"]), "side": d["posSide"], "price": float(d["bkPx"]), "size": float(d["sz"])}
            for group in payload.get("data", []) for d in group.get("details", [])]


def fetch_liquidations(get: Any, underlying: str, stop_ms: int) -> pd.DataFrame:
    """Newest first, until ``stop_ms``. Pages can repeat, so it stops when one adds nothing."""
    seen: dict[tuple[int, str, float, float], dict[str, Any]] = {}
    params = {"instType": "SWAP", "uly": underlying, "state": "filled", "limit": str(PAGE)}
    for _ in range(MAX_LIQUIDATION_PAGES):
        rows = parse_liquidations(get(LIQUIDATIONS_URL, params))
        fresh = [r for r in rows if (r["ts"], r["side"], r["price"], r["size"]) not in seen]
        if not fresh:
            break
        seen.update({(r["ts"], r["side"], r["price"], r["size"]): r for r in fresh})
        oldest = min(r["ts"] for r in rows)
        if oldest <= stop_ms:
            break
        params["after"] = str(oldest)
    return pd.DataFrame(list(seen.values()), columns=LIQUIDATION_KEY)


def http_get(url: str, params: dict[str, str]) -> dict[str, Any]:
    response = httpx.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if payload.get("code") not in ("0", 0):
        raise RuntimeError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
    return payload


def update_liquidations(client: Any, data_dir: Path, symbol: str, get: Any = http_get) -> pd.DataFrame:
    """Merge new liquidation orders into the cache, with their value in USDT."""
    path = liquidations_path(data_dir, symbol)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    stop = int(cached["ts"].max()) if len(cached) else 0
    client.load_markets()
    market = client.market(symbol)
    fresh = fetch_liquidations(get, f"{market['base']}-{market['quote']}", stop)
    fresh["usd"] = fresh["size"] * float(market["contractSize"]) * fresh["price"]
    merged = pd.concat([cached, fresh]) if len(cached) else fresh
    merged = merged.drop_duplicates(LIQUIDATION_KEY).sort_values("ts").reset_index(drop=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    log.info("%s liquidations: %d orders (+%d)", symbol, len(merged), len(merged) - len(cached))
    return merged


def hourly_liquidations(liquidations: pd.DataFrame) -> pd.DataFrame:
    """USDT liquidated per hour for each side: ``long`` and ``short`` columns."""
    if liquidations.empty:
        return pd.DataFrame(columns=["long", "short"], dtype=float)
    hours = pd.to_datetime(liquidations["ts"], unit="ms", utc=True).dt.floor("1h")
    table = liquidations.groupby([hours, "side"])["usd"].sum().unstack()
    return table.reindex(columns=["long", "short"]).fillna(0.0)
