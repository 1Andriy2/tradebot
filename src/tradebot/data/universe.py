"""Which coins the 5-minute paper lane trades: the busiest by number of trades in 24 hours.

OKX publishes volume but not a trade count, so the count comes from Binance's public tickers and
only coins with a live USDT swap on the trading venue are kept. If Binance cannot be reached the
list is ranked by the venue's own volume, and the file says so. The list is picked again once a
week. It is chosen from what is busy now and traded from then on, so it says nothing about the past:
a backtest on today's list would be testing coins already known to have moved.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx
import pandas as pd

from tradebot.store import Store

log = logging.getLogger(__name__)

BUCKET = "data"
FILE = "universe.json"
COINS = 10
PICK_EVERY = pd.Timedelta("7D")
TIMEOUT = 20.0
QUOTE = "USDT"
# Futures first, since that is what is traded here. Spot is the next best count, and its address
# is a separate public mirror that answers in places where the main one is blocked.
COUNT_SOURCES = {
    "binance_futures": "https://fapi.binance.com/fapi/v1/ticker/24hr",
    "binance_spot": "https://data-api.binance.vision/api/v3/ticker/24hr",
}
FALLBACK = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT"]
# Busy but flat: a coin pegged to the dollar has nothing for these strategies to trade.
PEGGED = {"USDC", "USD1", "FDUSD", "TUSD", "DAI", "USDE", "USDS", "PYUSD", "RLUSD", "USDG", "EUR", "EURI"}
RULES = 2  # raise when the choice changes, so a saved list is picked again instead of waiting a week
SCALED = re.compile(r"^1000+")  # Binance lists cheap coins as 1000PEPE, OKX as PEPE

Fetch = Callable[[str], Any]


def fetch_json(url: str) -> Any:
    response = httpx.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def trade_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Trades in 24 hours per coin. A coin listed twice keeps its larger count."""
    counts: dict[str, int] = {}
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if symbol.endswith(QUOTE) and int(row.get("count") or 0) > 0:
            base = SCALED.sub("", symbol.removesuffix(QUOTE))
            if base in PEGGED:
                continue
            counts[base] = max(counts.get(base, 0), int(row["count"]))
    return counts


def venue_swaps(client: Any) -> dict[str, str]:
    """Coin name to swap symbol, for the live USDT swaps of the venue."""
    return {m["base"]: m["symbol"] for m in client.load_markets().values()
            if m.get("swap") and m.get("linear") and m.get("quote") == QUOTE and m.get("active")}


def by_volume(client: Any, swaps: dict[str, str]) -> dict[str, int]:
    tickers = client.fetch_tickers(list(swaps.values()))
    return {coin: int(tickers[s].get("quoteVolume") or 0) for coin, s in swaps.items()
            if s in tickers and coin not in PEGGED}


def pick(client: Any, now: pd.Timestamp, fetch: Fetch = fetch_json) -> dict[str, Any]:
    swaps = venue_swaps(client)
    counts: dict[str, int] = {}
    source = ""
    for name, url in COUNT_SOURCES.items():
        try:
            counts = {c: n for c, n in trade_counts(fetch(url)).items() if c in swaps}
        except Exception as exc:
            log.warning("no trade counts from %s: %s", name, type(exc).__name__)
            continue
        if len(counts) >= COINS:
            source = name
            break
    if not source:
        counts, source = by_volume(client, swaps), "venue_volume"
    top = sorted(counts, key=lambda c: (-counts[c], c))[:COINS]
    return {"picked_at": now.isoformat(), "rules": RULES, "source": source,
            "coins": [{"symbol": swaps[c], "activity": counts[c]} for c in top]}


def read(store: Store) -> dict[str, Any] | None:
    try:
        saved: dict[str, Any] = json.loads(store.download(BUCKET, FILE))
    except Exception as exc:  # nothing saved yet, or storage is down: both mean "no list here"
        log.warning("no saved coin list: %s", type(exc).__name__)
        return None
    return saved


def current(store: Store, client: Any, now: pd.Timestamp, fetch: Fetch = fetch_json) -> list[str]:
    """The coins to trade. A failed pick keeps the old list rather than stopping the lane."""
    saved = read(store)
    if saved and saved.get("rules") == RULES and now - pd.Timestamp(saved["picked_at"]) < PICK_EVERY:
        return [c["symbol"] for c in saved["coins"]]
    try:
        fresh = pick(client, now, fetch)
        if not fresh["coins"]:
            raise RuntimeError("the pick found no coins")
        store.upload(BUCKET, FILE, json.dumps(fresh).encode())
    except Exception as exc:
        log.warning("coin list not picked: %s", type(exc).__name__)
        return [c["symbol"] for c in saved["coins"]] if saved else list(FALLBACK)
    coins = [c["symbol"] for c in fresh["coins"]]
    store.insert("events", [{"ts": now, "type": "info", "message": f"Новий список монет: {', '.join(coins)}",
                             "details": {"source": fresh["source"], "coins": fresh["coins"]}}])
    return coins
