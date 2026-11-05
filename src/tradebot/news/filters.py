"""Cheap checks that run before the model: what is worth scoring, and what repeats a known story.

The lists are written down in advance and name roles and institutions rather than people, so
they do not go stale when an office changes hands.
"""

from __future__ import annotations

import re
from typing import Any

CRYPTO_TERMS = (r"bitcoin|btc|ethereum|ether|crypto\w*|stablecoins?|blockchain|tether|usdt|usdc|"
                r"coinbase|binance|kraken|microstrategy|bitcoin miners?|spot etfs?")
POLICY_TERMS = (r"federal reserve|fed chair\w*|fomc|interest rates?|rate (cuts?|hikes?)|treasury secretary|"
                r"sec chair\w*|cftc|tariffs?|sanctions?|ofac|executive order|debt ceiling|"
                r"government shutdown|central bank")
MARKET_TERMS = r"stock market|wall street|s&p 500|nasdaq|dow jones|recession|inflation|cpi"
CONFLICT_TERMS = r"invasion|ceasefire|missile strikes?|airstrikes?|nuclear (strike|test|threat)"
SOFT_TERMS = r"hack\w*|exploit\w*|etfs?|regulat\w+|nominat\w+|appoint\w+|election\w*|war"

# Terms that are specific by themselves. Loud sources must pass this stricter list: GDELT because of
# its volume, and 8-K filings and rules because boilerplate such as "Regulation FD" or "Election of
# Directors" appears in nearly all of them and would match the soft terms.
STRICT_WATCH = re.compile(rf"\b({CRYPTO_TERMS}|{POLICY_TERMS}|{MARKET_TERMS}|{CONFLICT_TERMS})\b", re.I)
WATCH = re.compile(rf"\b({CRYPTO_TERMS}|{POLICY_TERMS}|{MARKET_TERMS}|{CONFLICT_TERMS}|{SOFT_TERMS})\b",
                   re.I)
STRICT_SOURCES = {"sec_8k", "fr_rules"}

# Not a security boundary: the model has no tools and its output is validated either way. This only
# keeps obvious attempts away from it. The idea comes from koala73/worldmonitor; the code is ours.
SUSPICIOUS = re.compile(r"(ignore|disregard|forget|override)\s+(all\s+|any\s+)?(the\s+|your\s+)?"
                        r"(previous|prior|above|earlier|system)\b|<\|[a-z_]+\|>|\[/?(inst|sys)\]|"
                        r"^\s*(system|assistant)\s*:", re.I)

# Few items a day, and a dull title here can still matter.
ALWAYS_SCORED = {"fed_press", "fed_speeches", "sec_press", "cftc", "white_house", "bls", "truth_trump",
                 "fr_president", "polymarket"}

# Posts with nothing to rate: a bare link or a media caption.
EMPTY_POST = re.compile(r"^(https?:/\S*(\s+\S+){0,2}|view (image|video))$", re.I)

SIMILAR = 0.6
WORD = re.compile(r"[a-z0-9$%]{3,}")


def relevant(item: dict[str, Any]) -> bool:
    if EMPTY_POST.match(item["title"]):
        return False
    if item["source"] in ALWAYS_SCORED:
        return True
    return bool((STRICT_WATCH if item["source"] in STRICT_SOURCES else WATCH).search(item["title"]))


def suspicious(title: str) -> bool:
    return bool(SUSPICIOUS.search(title))


def words(title: str) -> frozenset[str]:
    return frozenset(WORD.findall(title.lower()))


def same_story(a: frozenset[str], b: frozenset[str]) -> bool:
    """Jaccard overlap of the words in two headlines."""
    return bool(a and b) and len(a & b) / len(a | b) >= SIMILAR
