"""Free public sources of news and events, normalised to one item shape.

All text from here is untrusted. It is stored and shown to a model that has no tools, and it is
never executed. Each source fails on its own, so one broken feed does not stop the others.
Reddit is left out on purpose: its RSS ends on 2026-11-13 and its public API is closing.
"""

from __future__ import annotations

import email.utils
import html
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from tradebot.data.news import FEEDS, fetch_feed
from tradebot.news.filters import STRICT_WATCH, mentions

log = logging.getLogger(__name__)

Item = dict[str, Any]
Fetch = Callable[[str], bytes]
Source = Callable[[Fetch, pd.Timestamp], list[Item]]

OFFICIAL, NEWS, SOCIAL, GEO, MARKET, WATCH = "official", "news", "social", "geo", "market", "watch"

MAX_DOWNLOAD = 30_000_000  # a GDELT file is about 8 MB
MAX_UNZIPPED = 200_000_000
COLLECT_SECONDS = 120  # the job gets 10 minutes: leave room to score and save what was found
ASSET_SOURCES = ("google_", "yahoo_", "filings_", "history_")  # sources outside the general stream
BACKFILL_DAYS = 30  # how far back a new subscription starts; Google News answers with up to 100 items
WORKERS = 8  # downloads are independent, and the level should be ready before the bar closes
MAX_TEXT = 500
HISTORY = "history"  # the lane of general news from before the bot was listening
HISTORY_TICKER = "MARKET"
HISTORY_TOPICS = {
    "bitcoin": "bitcoin", "ethereum": "ethereum", "crypto_rules": "crypto regulation",
    "fed": '"Federal Reserve" rates', "sec": "SEC crypto", "inflation": "US inflation CPI",
    "stocks": '"stock market"', "tariffs": "tariffs trade",
}
ATOM = "{http://www.w3.org/2005/Atom}"

OFFICIAL_FEEDS = {
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
    "sec_press": "https://www.sec.gov/news/pressreleases.rss",
    "cftc": "https://www.cftc.gov/RSS/RSSGP/rssgp.xml",
    "white_house": "https://www.whitehouse.gov/presidential-actions/feed/",
    "bls": "https://www.bls.gov/feed/bls_latest.rss",
    "sec_8k": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&count=100&output=atom",
}
TRUTH_FEED = "https://trumpstruth.org/feed"  # independent archive; the official feed is paid
FEDERAL_REGISTER = "https://www.federalregister.gov/api/v1/documents.json"
GDELT_LATEST = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"
POLYMARKET_EVENTS = "https://gamma-api.polymarket.com/events"
POLYMARKET_TAGS = ("politics", "economy", "geopolitics")  # the top of the unfiltered list is sport
MARKET_MOVE = 0.10  # one-hour change in probability
MARKET_VOLUME = 50_000.0
PRICE_MARKET = re.compile(r"\b(price|above|below|up or down|hit \$|reach \$)", re.I)

GOOGLE_NEWS = "https://news.google.com/rss/search"
YAHOO_NEWS = "https://feeds.finance.yahoo.com/rss/2.0/headline"
SEC_COMPANY = "https://www.sec.gov/cgi-bin/browse-edgar"
# Legal forms, and the descriptive words headlines leave out: 'Palantir', not 'Palantir Technologies'.
COMPANY_WORDS = ("inc|incorporated|corp|corporation|co|com|company|ltd|limited|plc|holdings?|group|"
                 "n\\.?v|s\\.?a|ag|se|class [a-z]|technologies|technology|platforms|systems|motor")
COMPANY_SUFFIX = re.compile(rf"[\s,&]+(?:{COMPANY_WORDS})\.?$|\.com$|\s*/[a-z]{{2}}/?$|[\s,&/]+$", re.I)

TAG = re.compile(r"<[^>]+>")
INVISIBLE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁠-⁩﻿]")
PLACEHOLDER = re.compile(r"^(image |video )?post from ", re.I)
TELEGRAM_POST = re.compile(r'data-post="([\w/]+)"')
TELEGRAM_TIME = re.compile(r'<time datetime="([^"]+)"')
TELEGRAM_TEXT = re.compile(r'tgme_widget_message_text[^>]*>(.*?)</div>', re.S)


def clean(text: str | None) -> str:
    """Plain one-line text: no tags, entities, control or invisible characters."""
    plain = INVISIBLE.sub("", html.unescape(TAG.sub(" ", text or "")))
    return " ".join(plain.split())[:MAX_TEXT]


def parse_time(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        stamp = pd.Timestamp(email.utils.parsedate_to_datetime(value))
    except (TypeError, ValueError):
        try:
            stamp = pd.Timestamp(value.strip())
        except ValueError:
            return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def make_item(published: pd.Timestamp | None, seen_at: pd.Timestamp, source: str, kind: str,
              text: str | None, url: str | None) -> Item | None:
    """One item, or None if it lacks a date, text or a web link."""
    title, link = clean(text), (url or "").strip()
    if published is None or not title or not link.startswith(("https://", "http://")):
        return None
    return {"published": published, "first_seen": seen_at, "source": source, "kind": kind,
            "title": title, "url": link}


def parse_feed(content: bytes, source: str, kind: str, seen_at: pd.Timestamp) -> list[Item]:
    """RSS items and Atom entries. Anything malformed is skipped."""
    root = ET.fromstring(content)
    rows: list[Item | None] = []
    for entry in root.iter("item"):
        text = entry.findtext("title")
        if not clean(text) or PLACEHOLDER.match(clean(text)):
            text = entry.findtext("description")
        rows.append(make_item(parse_time(entry.findtext("pubDate")), seen_at, source, kind, text,
                              entry.findtext("link")))
    for entry in root.iter(f"{ATOM}entry"):
        link = entry.find(f"{ATOM}link")
        text = f"{entry.findtext(f'{ATOM}title') or ''} {entry.findtext(f'{ATOM}summary') or ''}"
        rows.append(make_item(parse_time(entry.findtext(f"{ATOM}updated")), seen_at, source, kind, text,
                              link.get("href") if link is not None else None))
    return [r for r in rows if r]


def parse_federal_register(content: bytes, source: str, seen_at: pd.Timestamp) -> list[Item]:
    """These carry a date with no time, and midnight would make today's document look half a day
    old. The query looks back only a day, so the time of first sight is close enough."""
    rows = [make_item(seen_at, seen_at, source, OFFICIAL, d.get("title"), d.get("html_url"))
            for d in json.loads(content)["results"]]
    return [r for r in rows if r]


def parse_gkg(content: bytes, seen_at: pd.Timestamp) -> list[Item]:
    """Titles from one GDELT 15-minute file that match the watch list.

    The search API rate-limits shared addresses at once, so the raw files are used instead.
    Each domain is its own source, which is what lets several outlets confirm one story.
    """
    rows: list[Item | None] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        info = archive.infolist()[0]
        if info.file_size > MAX_UNZIPPED:
            raise ValueError(f"GDELT file unpacks to {info.file_size} bytes")
        with archive.open(info) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                cols = line.rstrip("\n").split("\t")
                found = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", cols[26]) if len(cols) > 26 else None
                if found and STRICT_WATCH.search(found.group(1)):
                    rows.append(make_item(parse_time(cols[1]), seen_at, f"gdelt:{cols[3]}", GEO,
                                          found.group(1), cols[4]))
    return [r for r in rows if r]


def parse_polymarket(content: bytes, seen_at: pd.Timestamp) -> list[Item]:
    """Large one-hour moves in political and economic markets, at most one per market per day."""
    rows: list[Item | None] = []
    for event in json.loads(content):
        for market in event.get("markets") or []:
            change = float(market.get("oneHourPriceChange") or 0)
            volume = float(market.get("volume24hr") or 0)
            question = clean(market.get("question"))
            if abs(change) < MARKET_MOVE or volume < MARKET_VOLUME or PRICE_MARKET.search(question) \
                    or not market.get("outcomePrices"):
                continue
            yes = float(json.loads(market["outcomePrices"])[0])
            url = f"https://polymarket.com/event/{event['slug']}?m={market['id']}&d={seen_at:%Y-%m-%d}"
            text = f"Prediction market: '{question}' moved {change:+.0%} in an hour to {yes:.0%}"
            rows.append(make_item(seen_at, seen_at, "polymarket", MARKET, text, url))
    return [r for r in rows if r]


def parse_telegram(content: bytes, channel: str, seen_at: pd.Timestamp) -> list[Item]:
    """Posts from the public web preview of a channel; no account or key is needed."""
    rows: list[Item | None] = []
    for part in content.decode("utf-8", "replace").split("tgme_widget_message_wrap")[1:]:
        post, stamp = TELEGRAM_POST.search(part), TELEGRAM_TIME.search(part)
        text = TELEGRAM_TEXT.search(part)
        if post and stamp and text:
            rows.append(make_item(parse_time(stamp.group(1)), seen_at, f"telegram:{channel}", SOCIAL,
                                  text.group(1), f"https://t.me/{post.group(1)}"))
    return [r for r in rows if r]


def search_name(asset: Item) -> str:
    """The name news uses: 'The Boeing Company' is written 'Boeing'. Measured on one day
    of Google News, a ticker alone found 0 to 44% relevant headlines for SOL, NEAR, LINK and T; the
    name found 41 to 98%. Only stocks are trimmed, since a coin's name is already the plain one."""
    name = str(asset["name"]).strip()
    if asset.get("kind") == "crypto":
        return name
    while (shorter := COMPANY_SUFFIX.sub("", name)) != name and shorter:
        name = shorter
    return re.sub(r"^the\s+", "", name, flags=re.I)


def needs_backfill(asset: Item) -> bool:
    """History is fetched once. A run forgets links after a few days, so asking for a month on every
    run would bring the same old stories back to the model again and again."""
    return not asset.get("backfilled_at")


def _for_asset(rows: list[Item], asset: Item) -> list[Item]:
    """Tag items with the asset, keeping only headlines that really name it."""
    name, ticker, backfill = search_name(asset), str(asset["ticker"]), needs_backfill(asset)
    return [{**r, "kind": WATCH, "watch": asset["id"], "ticker": ticker, "backfill": backfill} for r in rows
            if mentions(r["title"], name, ticker)]


def parse_google(content: bytes, seen_at: pd.Timestamp) -> list[Item]:
    """Each item names its outlet, which becomes the source, so outlets can confirm one another."""
    rows: list[Item | None] = []
    for entry in ET.fromstring(content).iter("item"):
        outlet = clean(entry.findtext("source"))
        title = clean(entry.findtext("title")).removesuffix(f" - {outlet}")
        rows.append(make_item(parse_time(entry.findtext("pubDate")), seen_at, f"google:{outlet}", WATCH,
                              title, entry.findtext("link")))
    return [r for r in rows if r]


def _google(asset: Item) -> Source:
    words = f'"{search_name(asset)}"' + (" crypto" if asset["kind"] == "crypto" else "")
    window = f"{BACKFILL_DAYS}d" if needs_backfill(asset) else "1d"
    query = urlencode({"q": f"{words} when:{window}", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return lambda fetch, now: _for_asset(parse_google(fetch(f"{GOOGLE_NEWS}?{query}"), now), asset)


def _yahoo(asset: Item) -> Source:
    symbol = f"{asset['ticker']}-USD" if asset["kind"] == "crypto" else asset["ticker"]
    query = urlencode({"s": symbol, "region": "US", "lang": "en-US"})
    url = f"{YAHOO_NEWS}?{query}"
    return lambda fetch, now: _for_asset(parse_feed(fetch(url), "yahoo", WATCH, now), asset)


def _filings(asset: Item) -> Source:
    query = urlencode({"action": "getcompany", "CIK": asset["source_id"], "type": "8-K", "count": 10,
                       "output": "atom"})
    def source(fetch: Fetch, now: pd.Timestamp) -> list[Item]:
        rows = parse_feed(fetch(f"{SEC_COMPANY}?{query}"), "sec_8k", WATCH, now)
        named = [{**r, "title": f"{search_name(asset)}: {r['title']}"} for r in rows]  # the feed omits it
        return _for_asset(named, asset)
    return source


def _history(words: str) -> Source:
    query = urlencode({"q": f"{words} when:{BACKFILL_DAYS}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    tags = {"kind": WATCH, "watch": HISTORY, "ticker": HISTORY_TICKER, "backfill": True}
    return lambda fetch, now: [{**r, **tags} for r in parse_google(fetch(f"{GOOGLE_NEWS}?{query}"), now)]


def history_sources() -> dict[str, Source]:
    """A month of general news, fetched once. It is a followed lane in all but name: shown on the
    page, kept out of the risk level and the studies, because the model may remember what followed."""
    return {f"history_{topic}": _history(words) for topic, words in HISTORY_TOPICS.items()}


def asset_sources(watchlist: list[Item]) -> dict[str, Source]:
    """Sources for the assets being followed. Nothing is fetched for anything else."""
    sources: dict[str, Source] = {}
    for asset in watchlist:
        sources |= {f"google_{asset['id']}": _google(asset), f"yahoo_{asset['id']}": _yahoo(asset)}
        if asset["kind"] == "stock" and asset.get("source_id"):
            sources[f"filings_{asset['id']}"] = _filings(asset)
    return sources


def _feed(name: str, kind: str, url: str) -> Source:
    return lambda fetch, now: parse_feed(fetch(url), name, kind, now)


def _federal_register(name: str, doc_type: str) -> Source:
    def source(fetch: Fetch, now: pd.Timestamp) -> list[Item]:
        since = (now - pd.Timedelta("1D")).strftime("%Y-%m-%d")  # inside the news file's 48 hours
        query = urlencode({"per_page": 100, "order": "newest", "conditions[type][]": doc_type,
                           "conditions[publication_date][gte]": since})
        return parse_federal_register(fetch(f"{FEDERAL_REGISTER}?{query}"), name, now)
    return source


def _gdelt(fetch: Fetch, now: pd.Timestamp) -> list[Item]:
    lines = fetch(GDELT_LATEST).decode().split("\n")
    url = next(line.split()[2] for line in lines if ".gkg.csv.zip" in line)
    return parse_gkg(fetch(url.replace("http://", "https://")), now)


def _polymarket(tag: str) -> Source:
    query = urlencode({"limit": 50, "active": "true", "closed": "false", "tag_slug": tag,
                       "order": "volume24hr", "ascending": "false"})
    return lambda fetch, now: parse_polymarket(fetch(f"{POLYMARKET_EVENTS}?{query}"), now)


def _telegram(channel: str) -> Source:
    return lambda fetch, now: parse_telegram(fetch(f"https://t.me/s/{channel}"), channel, now)


def all_sources(channels: list[str], watchlist: list[Item] | None = None) -> dict[str, Source]:
    sources = asset_sources(watchlist or [])
    sources |= {name: _feed(name, NEWS, url) for name, url in FEEDS.items()}
    sources |= {name: _feed(name, OFFICIAL, url) for name, url in OFFICIAL_FEEDS.items()}
    sources |= {"truth_trump": _feed("truth_trump", SOCIAL, TRUTH_FEED), "gdelt": _gdelt,
                "fr_president": _federal_register("fr_president", "PRESDOCU"),
                "fr_rules": _federal_register("fr_rules", "RULE")}
    sources |= {f"polymarket_{tag}": _polymarket(tag) for tag in POLYMARKET_TAGS}
    return sources | {f"telegram_{c}": _telegram(c) for c in channels}


def download(url: str) -> bytes:
    return fetch_feed(url, MAX_DOWNLOAD)


def collect(sources: dict[str, Source], now: pd.Timestamp, fetch: Fetch = download,
            seconds: float = COLLECT_SECONDS) -> tuple[list[Item], dict[str, int]]:
    """Items from every source, and how many each gave (-1 for a source that failed or was skipped
    because time ran out)."""
    pool = ThreadPoolExecutor(WORKERS)
    futures = {name: pool.submit(source, fetch, now) for name, source in sources.items()}
    wait(futures.values(), timeout=max(seconds, 0))
    pool.shutdown(wait=False, cancel_futures=True)
    items: list[Item] = []
    counts: dict[str, int] = {}
    for name, future in futures.items():
        counts[name] = -1
        if not future.done() or future.cancelled():
            log.warning("news source %s skipped: out of time", name)
            continue
        try:
            found = future.result()
        except Exception as exc:  # outside data can fail in any way, and must stop only this source
            log.warning("news source %s failed: %s %s", name, type(exc).__name__, exc)
            continue
        counts[name] = len(found)
        items += found
    return items, counts
