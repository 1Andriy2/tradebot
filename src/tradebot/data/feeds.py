"""Live data beside the candles, for strategies that trade on liquidations and news.

Exchanges publish real liquidations for a short while only: OKX answers about the last 24 hours. So
these strategies cannot be tested on years of history, and they start as paper agents that collect
their own record. Open interest gives an estimate of where liquidations wait, made the way public
heatmaps make theirs: nobody publishes the real levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradebot.data.market_stats import _paged, fetch_liquidations, http_get

LIQUIDATION_HOURS = 25
OPEN_INTEREST_DAYS = 7
HOUR = pd.Timedelta("1h")


class FeedMissing(RuntimeError):
    """The data this strategy needs could not be read this step. It waits; it does not trade blind."""


@dataclass
class Feed:
    liquidations: pd.DataFrame | None = None  # USDT liquidated per bar: long, short
    open_interest: pd.Series | None = None  # hourly, USDT
    news: pd.DataFrame | None = None  # this coin's news levels, oldest first


def liquidation_bars(client: Any, symbol: str, timeframe: str, get: Any = http_get) -> pd.DataFrame:
    """Liquidated value per bar and side, labelled by the bar's open like the candles."""
    client.load_markets()
    market = client.market(symbol)
    now = int(client.milliseconds())
    rows = fetch_liquidations(get, f"{market['base']}-{market['quote']}", now - LIQUIDATION_HOURS * 3_600_000)
    if rows.empty:
        return pd.DataFrame(columns=["long", "short"], dtype=float)
    usd = rows["size"] * float(market["contractSize"]) * rows["price"]
    step = pd.Timedelta(client.parse_timeframe(timeframe), unit="s")
    bars = pd.to_datetime(rows["ts"], unit="ms", utc=True).dt.floor(step)
    table = usd.groupby([bars, rows["side"]]).sum().unstack()
    return table.reindex(columns=["long", "short"]).fillna(0.0)


def open_interest(client: Any, symbol: str) -> pd.Series:
    now = int(client.milliseconds())
    rows = _paged(client.fetch_open_interest_history, symbol, now - OPEN_INTEREST_DAYS * 86_400_000, now)
    values = {int(r["timestamp"]): float(r["openInterestValue"]) for r in rows}
    series = pd.Series(values, dtype=float)
    series.index = pd.to_datetime(series.index, unit="ms", utc=True)
    return series.sort_index()
