"""Hard guard: authenticated traffic may only reach demo endpoints.

An endpoint is a host plus, for venues whose demo shares the production host (OKX), a
mandatory demo header. The allowlist is a code constant on purpose: it cannot be extended
from config or env. There is no code path for live trading in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class DemoEndpoint:
    hosts: frozenset[str]
    required_headers: dict[str, str] = field(default_factory=dict)


DEMO_ENDPOINTS: dict[str, DemoEndpoint] = {
    "bybit": DemoEndpoint(frozenset({"api-demo.bybit.com", "stream-demo.bybit.com"})),
    # OKX demo uses the production host; the x-simulated-trading header selects demo, and
    # demo API keys are rejected by the live environment.
    "okx": DemoEndpoint(frozenset({"www.okx.com", "openapi.okx.com"}), {"x-simulated-trading": "1"}),
}


class LiveEndpointError(RuntimeError):
    """Raised when an authenticated client would reach a non-demo endpoint."""


def _endpoint(exchange_id: str) -> DemoEndpoint:
    ep = DEMO_ENDPOINTS.get(exchange_id)
    if ep is None:
        raise LiveEndpointError(f"venue '{exchange_id}' has no demo allowlist; refusing to run")
    return ep


def assert_demo_url(exchange_id: str, url: str) -> None:
    host = (urlparse(url).hostname or "").lower()
    if host not in _endpoint(exchange_id).hosts:
        raise LiveEndpointError(
            f"endpoint host '{host}' is not on the demo allowlist for '{exchange_id}'"
        )


def assert_demo_headers(exchange_id: str, headers: dict[str, Any] | None) -> None:
    lowered = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    for key, value in _endpoint(exchange_id).required_headers.items():
        if lowered.get(key) != value:
            raise LiveEndpointError(f"demo header '{key}: {value}' missing for '{exchange_id}'")


def _iter_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [u for v in value.values() for u in _iter_urls(v)]
    return []


def install_guard(client: Any, exchange_id: str) -> None:
    """Check the configured API urls/headers now, and every future request.

    Works with ccxt clients: wraps the instance's ``fetch`` method, through which every
    REST call passes after signing. Headers are checked as ccxt will send them
    (instance headers merged with per-request headers).
    """
    hostname = getattr(client, "hostname", "") or ""
    for url in _iter_urls(client.urls.get("api")):
        assert_demo_url(exchange_id, url.replace("{hostname}", hostname))
    assert_demo_headers(exchange_id, client.headers)

    original_fetch = client.fetch

    def guarded_fetch(url: str, method: str = "GET", headers: dict[str, Any] | None = None,
                      body: Any = None) -> Any:
        assert_demo_url(exchange_id, url)
        assert_demo_headers(exchange_id, {**(client.headers or {}), **(headers or {})})
        return original_fetch(url, method, headers, body)

    client.fetch = guarded_fetch
