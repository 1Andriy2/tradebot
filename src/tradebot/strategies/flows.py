"""Strategies that trade on news and on liquidations.

None of these can be tested on years of history: OKX shows liquidations for about a day, and news is
rated as it arrives. So they start as paper agents, and every rule and number here was fixed by
reasoning before any result was seen.

News (rated by the news job, see tradebot.news.risk):
- ``NewsEach`` trades each strong new story in its direction, for a few hours.
- ``NewsBalance`` trades the balance of recent stories, weighed by impact and faded by age, so a
  good story can end a fall that bad news started.

Liquidations (real orders from OKX, and an estimated map from open interest):
- ``LiqFade`` waits for a burst of liquidations and takes the other side: the forced sellers are gone.
- ``LiqSweep`` looks past the burst: with more liquidations waiting the same way it expects them to be
  taken too, otherwise it takes the other side.
- ``LiqMagnet`` leans toward the side where more estimated liquidations wait nearby.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tradebot.data.feeds import HOUR, Feed, FeedMissing
from tradebot.ml.features import resample_ohlcv
from tradebot.strategies.base import Strategy

MARGIN = pd.Timedelta("1min")  # a news level is stamped just before its file is uploaded
LEVEL_AGE = pd.Timedelta("1h")  # an older balance is not trusted, as in the news filter
# Leverage of new positions, and how much of the open interest takes each: a fixed guess.
LEVERAGE = {10.0: 0.4, 25.0: 0.35, 50.0: 0.25}
MAINTENANCE = 0.005


class FeedStrategy(Strategy):
    needs: tuple[str, ...] = ()

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.feed: Feed | None = None

    def _need(self) -> Feed:
        if self.feed is None:
            raise FeedMissing(self.name)
        return self.feed


def _bar(df: pd.DataFrame) -> pd.Timedelta:
    return pd.Timedelta(pd.Series(df.index).diff().median())


def _hold(events: pd.Series, bars: int) -> pd.Series:
    """Each event's sign held for ``bars`` bars, a newer event replacing an older one."""
    marked = events.where(events != 0)
    return marked.ffill(limit=bars - 1).fillna(0.0)


def _news_at_bars(df: pd.DataFrame, news: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each level row placed on the bar whose close is the first after it was written."""
    if news.empty or column not in news:
        return pd.DataFrame(columns=[column])
    rows = news[["ts", column]].dropna()
    closes = pd.DatetimeIndex(pd.to_datetime(rows["ts"], utc=True) + MARGIN)
    bar = _bar(df)
    labels = (closes - bar).ceil(bar)
    return pd.DataFrame({column: rows[column].to_numpy(float), "ts": closes}, index=labels)


class NewsEach(FeedStrategy):
    name = "newsflow_each"
    needs = ("news",)

    def __init__(self, threshold: float = 0.4, hold_bars: int = 16) -> None:
        super().__init__(threshold=threshold, hold_bars=hold_bars)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        news = self._need().news
        placed = _news_at_bars(df, news if news is not None else pd.DataFrame(), "fresh")
        strongest = placed["fresh"].groupby(level=0).agg(lambda v: v.iloc[np.argmax(np.abs(v.to_numpy()))]) \
            if len(placed) else pd.Series(dtype=float)
        events = np.sign(strongest.where(strongest.abs() >= self.params["threshold"], 0.0))
        return _hold(events.reindex(df.index, fill_value=0.0), self.params["hold_bars"])


class NewsBalance(FeedStrategy):
    name = "newsflow_balance"
    needs = ("news",)

    def __init__(self, enter: float = 0.7, leave: float = 0.25) -> None:
        super().__init__(enter=enter, leave=leave)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        news = self._need().news
        placed = _news_at_bars(df, news if news is not None else pd.DataFrame(), "net")
        latest = placed.groupby(level=0).last() if len(placed) else placed
        net = latest["net"].reindex(df.index).ffill()
        seen = latest["ts"].reindex(df.index).ffill() if len(latest) else pd.Series(pd.NaT, index=df.index)
        fresh = (pd.DatetimeIndex(df.index) + _bar(df) - pd.to_datetime(seen, utc=True)) <= LEVEL_AGE
        balance = net.where(fresh, 0.0).fillna(0.0).to_numpy()
        out, pos = np.zeros(len(df)), 0.0
        for i, value in enumerate(balance):
            if pos != 0.0 and (abs(value) < self.params["leave"] or np.sign(value) != pos):
                pos = 0.0
            if abs(value) >= self.params["enter"]:
                pos = float(np.sign(value))
            out[i] = pos
        return pd.Series(out, index=df.index)


def bursts(df: pd.DataFrame, liquidations: pd.DataFrame, share: float, min_usd: float) -> pd.Series:
    """+1 where longs were wiped out in a burst, -1 where shorts were, 0 elsewhere. A burst is a bar
    holding at least ``share`` of the day's liquidations, and at least ``min_usd``."""
    table = liquidations.reindex(df.index, fill_value=0.0)
    day = float(liquidations[["long", "short"]].to_numpy().sum())
    floor = max(min_usd, share * day)
    longs, shorts = table["long"].to_numpy(float), table["short"].to_numpy(float)
    side = np.where((longs >= floor) & (longs >= shorts), 1.0, np.where(shorts >= floor, -1.0, 0.0))
    return pd.Series(side, index=df.index)


def liquidation_map(df: pd.DataFrame, oi: pd.Series, band: float, sweep: float) -> pd.DataFrame:
    """Estimated liquidations waiting near each bar's close.

    Every rise in open interest opens equal long and short value at that hour's price, spread over
    ``LEVERAGE``; each gets a liquidation price. A fall closes all of them in proportion. A level
    the price has crossed has been liquidated and is removed. An open-interest hour is used only
    once it is over, so nothing is seen early. Columns: value waiting above (shorts) and below
    (longs) within ``band``, weighed toward the nearest; and the share of all longs (shorts) waiting
    within ``sweep`` below (above).
    """
    cols = ["above", "below", "longs_close", "shorts_close"]
    out = pd.DataFrame(0.0, index=df.index, columns=cols)
    if oi.empty:
        return out
    hourly = resample_ohlcv(df, "1h")
    prices: npt.NDArray[Any] = np.empty(0)
    masses: npt.NDArray[Any] = np.empty(0)
    sides: npt.NDArray[Any] = np.empty(0)  # +1 a long's liquidation (below), -1 a short's (above)
    closes = pd.DatetimeIndex(df.index) + _bar(df)
    prev = None
    order = np.searchsorted(closes, pd.DatetimeIndex(oi.index) + HOUR)
    known_from = {stamp: int(at) for stamp, at in zip(oi.index, order, strict=True)}
    feature_rows = np.zeros((len(df), 4))
    stamps = list(oi.index)
    for n, h in enumerate(stamps):
        if h not in hourly.index:
            prev = float(oi[h])
            continue
        candle = hourly.loc[h]
        alive = ~(((sides > 0) & (prices >= candle["low"])) | ((sides < 0) & (prices <= candle["high"])))
        prices, masses, sides = prices[alive], masses[alive], sides[alive]
        value = float(oi[h])
        if prev is not None and value > prev:
            add = value - prev
            price = float(candle["close"])
            for lev, weight in LEVERAGE.items():
                long_at, short_at = price * (1 - 1 / lev + MAINTENANCE), price * (1 + 1 / lev - MAINTENANCE)
                prices = np.append(prices, [long_at, short_at])
                masses = np.append(masses, [add * weight, add * weight])
                sides = np.append(sides, [1.0, -1.0])
        elif prev is not None and prev > 0 and value < prev:
            masses = masses * value / prev
        prev = value
        start = known_from[h]
        end = known_from[stamps[n + 1]] if n + 1 < len(stamps) else len(df)
        for i in range(start, end):
            close = float(df["close"].iloc[i])
            dist = np.abs(prices / close - 1)
            near = 1 / (dist + 0.002)
            up = (sides < 0) & (prices > close) & (dist <= band)
            down = (sides > 0) & (prices < close) & (dist <= band)
            long_all, short_all = masses[sides > 0].sum(), masses[sides < 0].sum()
            longs_close = masses[down & (dist <= sweep)].sum() / long_all if long_all else 0.0
            shorts_close = masses[up & (dist <= sweep)].sum() / short_all if short_all else 0.0
            weighed = masses * near
            feature_rows[i] = [weighed[up].sum(), weighed[down].sum(), longs_close, shorts_close]
    out.loc[:, cols] = feature_rows
    return out


class LiqFade(FeedStrategy):
    name = "liq_fade"
    needs = ("liquidations",)

    def __init__(self, share: float = 0.15, min_usd: float = 25_000.0, hold_bars: int = 8) -> None:
        super().__init__(share=share, min_usd=min_usd, hold_bars=hold_bars)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        liq = self._need().liquidations
        if liq is None or liq.empty:
            return pd.Series(0.0, index=df.index)
        p = self.params
        return _hold(bursts(df, liq, p["share"], p["min_usd"]), p["hold_bars"])


class LiqSweep(FeedStrategy):
    name = "liq_sweep"
    needs = ("liquidations", "open_interest")

    def __init__(self, share: float = 0.15, min_usd: float = 25_000.0, hold_bars: int = 8,
                 sweep: float = 0.02, more: float = 0.3) -> None:
        super().__init__(share=share, min_usd=min_usd, hold_bars=hold_bars, sweep=sweep, more=more)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        feed = self._need()
        if feed.liquidations is None or feed.liquidations.empty or feed.open_interest is None:
            return pd.Series(0.0, index=df.index)
        p = self.params
        burst = bursts(df, feed.liquidations, p["share"], p["min_usd"])
        waiting = liquidation_map(df, feed.open_interest, p["sweep"], p["sweep"])
        # Longs wiped out (+1): more longs waiting just below means the fall goes on to take them.
        more = np.where(burst > 0, waiting["longs_close"] >= p["more"], waiting["shorts_close"] >= p["more"])
        events = np.where(more, -burst, burst)
        return _hold(pd.Series(events, index=df.index), p["hold_bars"])


class LiqMagnet(FeedStrategy):
    name = "liq_magnet"
    needs = ("open_interest",)

    def __init__(self, band: float = 0.03, ratio: float = 1.5) -> None:
        super().__init__(band=band, ratio=ratio)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        oi = self._need().open_interest
        if oi is None or oi.empty:
            return pd.Series(0.0, index=df.index)
        waiting = liquidation_map(df, oi, self.params["band"], self.params["band"])
        up, down, k = waiting["above"].to_numpy(), waiting["below"].to_numpy(), self.params["ratio"]
        side = np.where((up > 0) & (up >= k * down), 1.0, np.where((down > 0) & (down >= k * up), -1.0, 0.0))
        return pd.Series(side, index=df.index)
