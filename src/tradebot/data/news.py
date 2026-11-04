"""Crypto news headlines from public RSS feeds, kept in a Parquet cache.

Feeds hold only about a day of items and archives with reliable timestamps are not free, so
history starts when collection starts. Feed text is untrusted: it is stored and counted,
never executed or sent anywhere. Tone is a plain word count, written down in advance, so it
can be tested honestly. It needs no paid model and it does not understand the news.
"""

from __future__ import annotations

import email.utils
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

log = logging.getLogger(__name__)

FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "theblock": "https://www.theblock.co/rss.xml",
}
MAX_FEED_BYTES = 5_000_000
MAX_TITLE = 300
COLUMNS = ["published", "first_seen", "source", "title", "url", "btc", "eth", "tone"]

COINS = {"btc": re.compile(r"\b(bitcoin|btc)\b", re.I), "eth": re.compile(r"\b(ethereum|ether|eth)\b", re.I)}
POSITIVE = re.compile(r"\b(approv\w*|adopt\w*|inflows?|rall\w+|surg\w+|soar\w*|record high|upgrade\w*|"
                      r"partnership\w*|rebounds?|gains?)\b", re.I)
NEGATIVE = re.compile(r"\b(hack\w*|exploit\w*|bans?|banned|banning|lawsuits?|sues?|sued|crash\w*|"
                      r"outflows?|bankrupt\w*|plung\w+|fraud\w*|liquidat\w+|sell-?off\w*|drops?|falls?)\b", re.I)


def news_path(data_dir: Path) -> Path:
    return data_dir / "news.parquet"


def tone(title: str) -> int:
    """Positive minus negative words: above zero reads good, below zero reads bad."""
    return len(POSITIVE.findall(title)) - len(NEGATIVE.findall(title))


def parse_feed(content: bytes, source: str, seen_at: pd.Timestamp) -> list[dict[str, Any]]:
    """Items with a usable date, title and link. Anything malformed is skipped."""
    rows = []
    for item in ET.fromstring(content).iter("item"):
        title = " ".join((item.findtext("title") or "").split())[:MAX_TITLE]
        url = (item.findtext("link") or "").strip()
        date = item.findtext("pubDate")
        if not (title and url.startswith("https://") and date):
            continue
        try:
            published = pd.Timestamp(email.utils.parsedate_to_datetime(date)).tz_convert("UTC")
        except (TypeError, ValueError):
            continue
        rows.append({"published": published, "first_seen": seen_at, "source": source, "title": title,
                     "url": url, **{c: bool(p.search(title)) for c, p in COINS.items()},
                     "tone": tone(title)})
    return rows


def fetch_feed(url: str) -> bytes:
    with httpx.stream("GET", url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "tradebot-research/1.0"}) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_FEED_BYTES:
                raise ValueError(f"feed larger than {MAX_FEED_BYTES} bytes: {url}")
            chunks.append(chunk)
    return b"".join(chunks)


def update_news(data_dir: Path, now: pd.Timestamp, fetch: Any = fetch_feed) -> pd.DataFrame:
    """Add new items to the cache. One broken feed does not affect the others."""
    path = news_path(data_dir)
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    rows: list[dict[str, Any]] = []
    for source, url in FEEDS.items():
        try:
            rows += parse_feed(fetch(url), source, now)
        except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
            log.warning("news feed %s failed: %s", source, exc)
    fresh = pd.DataFrame(rows, columns=COLUMNS)
    merged = fresh if cached.empty else pd.concat([cached, fresh]) if len(fresh) else cached
    merged = merged.drop_duplicates("url", keep="first").sort_values("published").reset_index(drop=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path)
    log.info("news: %d items (+%d)", len(merged), len(merged) - len(cached))
    return merged
