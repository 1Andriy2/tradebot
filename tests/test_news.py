from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from tradebot.data import news as nw
from tradebot.data.sync import _paths
from tradebot.research.events import news_studies

NOW = pd.Timestamp("2026-11-04 12:00", tz="UTC")


def feed(*items: str) -> bytes:
    return f"<rss><channel>{''.join(items)}</channel></rss>".encode()


def item(title: str = "Bitcoin ETF approved", url: str = "https://x.example/1",
         date: str = "Wed, 04 Nov 2026 10:30:00 +0200") -> str:
    return f"<item><title>{title}</title><link>{url}</link><pubDate>{date}</pubDate></item>"


def test_tone_counts_words_written_in_advance() -> None:
    assert nw.tone("Bitcoin ETF approved as inflows surge") == 3
    assert nw.tone("Exchange hacked, users sue") == -2
    assert nw.tone("Ethereum developers meet on Tuesday") == 0
    assert nw.tone("Banks and banana prices") == 0  # "ban" must match a whole word
    assert nw.tone("China bans mining") == -1


def test_parse_feed_normalises_time_and_tags_coins() -> None:
    (row,) = nw.parse_feed(feed(item("Ethereum and BTC rally")), "x", NOW)
    assert row["published"] == pd.Timestamp("2026-11-04 08:30", tz="UTC")
    assert row["btc"] and row["eth"] and row["tone"] == 1 and row["first_seen"] == NOW


def test_parse_feed_skips_bad_items() -> None:
    rows = nw.parse_feed(feed(item(date="not a date"), item(url="javascript:alert(1)"), item(title=""),
                              item("Fine", "https://x.example/2")), "x", NOW)
    assert [r["title"] for r in rows] == ["Fine"]


def test_titles_are_trimmed() -> None:
    (row,) = nw.parse_feed(feed(item("A  \n  b" + "c" * 900)), "x", NOW)
    assert row["title"].startswith("A b") and len(row["title"]) == nw.MAX_TITLE


def test_cache_keeps_the_first_sighting(tmp_path: Path) -> None:
    def fetch(url: str) -> bytes:
        return feed(item())

    first = nw.update_news(tmp_path, NOW, fetch)
    assert len(first) == 1
    second = nw.update_news(tmp_path, NOW + pd.Timedelta("1h"), fetch)
    assert len(second) == 1 and second["first_seen"].iloc[0] == NOW


def test_one_broken_feed_does_not_stop_the_rest(tmp_path: Path) -> None:
    def fetch(url: str) -> bytes:
        if "coindesk" in url:
            raise httpx.ConnectError("down")
        if "decrypt" in url:
            return b"<not xml"
        return feed(item(url=f"https://x.example/{len(url)}"))

    assert len(nw.update_news(tmp_path, NOW, fetch)) >= 1


def test_all_feeds_down_keeps_the_cache(tmp_path: Path) -> None:
    nw.update_news(tmp_path, NOW, lambda url: feed(item()))

    def down(url: str) -> bytes:
        raise httpx.ConnectError("down")

    assert len(nw.update_news(tmp_path, NOW, down)) == 1


def test_xml_entity_bomb_is_rejected_or_harmless() -> None:
    bomb = b'<!DOCTYPE r [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]><rss><item>' \
           b"<title>&b;</title><link>https://x.example/1</link>" \
           b"<pubDate>Wed, 04 Nov 2026 10:30:00 +0000</pubDate></item></rss>"
    rows = nw.parse_feed(bomb, "x", NOW)
    assert all(len(r["title"]) <= nw.MAX_TITLE for r in rows)


def test_news_is_synced(tmp_path: Path) -> None:
    assert nw.news_path(tmp_path) in _paths(tmp_path, ["BTC/USDT:USDT"], [])


def test_news_studies_without_data() -> None:
    close = pd.Series(100.0, index=pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC"))
    rows = news_studies(close, pd.DataFrame(), "btc")
    assert len(rows) == 4 and all(r["status"] == "not_enough_data" for r in rows)


def test_news_studies_use_the_hour_after_publication() -> None:
    """A headline at 10:30 is first usable at the 11:00 close: the candle labelled 10:00."""
    idx = pd.date_range("2026-01-01", periods=3000, freq="1h", tz="UTC")
    r = np.random.default_rng(1).normal(0, 0.003, 3000)
    stamps = idx[50:2900:50] + pd.Timedelta("30min")
    for t in stamps:
        i = idx.get_loc(t.floor("1h"))
        r[i + 1: i + 5] -= 0.006
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    news = pd.DataFrame({"published": stamps, "btc": True, "eth": False, "tone": -1})
    studies = news_studies(close, news, "btc")
    bad = next(s for s in studies if s["test"] == "bad_news" and s["horizon_hours"] == 4)
    assert bad["status"] == "tested" and bad["n"] >= 30 and bad["mean_after"] < bad["base_mean"]
    assert bad["p_value"] < 0.01
