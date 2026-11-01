"""Hard guard: authenticated traffic may only reach demo/testnet endpoints.

The allowlist is a code constant on purpose. It cannot be extended from config or env.
There is no code path for live trading in this project.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

# venue id -> hosts that an authenticated client may talk to
DEMO_HOSTS: dict[str, frozenset[str]] = {
    "bybit": frozenset({"api-demo.bybit.com", "stream-demo.bybit.com"}),
}


class LiveEndpointError(RuntimeError):
    """Raised when an authenticated client would reach a non-demo endpoint."""


def assert_demo_url(exchange_id: str, url: str) -> None:
    allowed = DEMO_HOSTS.get(exchange_id)
    if allowed is None:
        raise LiveEndpointError(f"venue '{exchange_id}' has no demo allowlist; refusing to run")
    host = (urlparse(url).hostname or "").lower()
    if host not in allowed:
        raise LiveEndpointError(
            f"endpoint host '{host}' is not on the demo allowlist for '{exchange_id}'"
        )


def _iter_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [u for v in value.values() for u in _iter_urls(v)]
    return []


def install_guard(client: Any, exchange_id: str) -> None:
    """Check the configured API urls now, and host-check every future request.

    Works with ccxt clients: wraps the instance's ``fetch`` method, through which every
    REST call passes after signing.
    """
    hostname = getattr(client, "hostname", "") or ""
    for url in _iter_urls(client.urls.get("api")):
        assert_demo_url(exchange_id, url.replace("{hostname}", hostname))

    original_fetch = client.fetch

    def guarded_fetch(url: str, *args: Any, **kwargs: Any) -> Any:
        assert_demo_url(exchange_id, url)
        return original_fetch(url, *args, **kwargs)

    client.fetch = guarded_fetch
