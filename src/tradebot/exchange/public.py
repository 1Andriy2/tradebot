"""Unauthenticated market-data client. It never holds API keys, so it cannot trade."""

from __future__ import annotations

from typing import Any

import ccxt


def make_public_client(exchange_id: str) -> Any:
    cls = getattr(ccxt, exchange_id)
    client = cls({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    if client.apiKey or client.secret:
        raise RuntimeError("public client must not carry credentials")
    return client
