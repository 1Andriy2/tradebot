from __future__ import annotations

import ccxt
import pytest

from tradebot.exchange.guard import LiveEndpointError, assert_demo_url, install_guard
from tradebot.exchange.public import make_public_client


def test_demo_host_allowed() -> None:
    assert_demo_url("bybit", "https://api-demo.bybit.com/v5/order/create")


@pytest.mark.parametrize(
    "url",
    [
        "https://api.bybit.com/v5/order/create",
        "https://api-testnet.bybit.com/v5/order/create",
        "https://api-demo.bybit.com.evil.example/v5",
        "https://api.bytick.com/v5",
    ],
)
def test_non_demo_hosts_rejected(url: str) -> None:
    with pytest.raises(LiveEndpointError):
        assert_demo_url("bybit", url)


def test_unknown_venue_rejected() -> None:
    with pytest.raises(LiveEndpointError):
        assert_demo_url("binance", "https://api-demo.bybit.com")


def test_install_guard_crashes_on_live_client() -> None:
    client = ccxt.bybit({"apiKey": "x", "secret": "y"})
    with pytest.raises(LiveEndpointError):
        install_guard(client, "bybit")


def test_install_guard_accepts_demo_and_blocks_redirected_requests() -> None:
    client = ccxt.bybit({"apiKey": "x", "secret": "y"})
    client.enable_demo_trading(True)
    install_guard(client, "bybit")
    with pytest.raises(LiveEndpointError):
        client.fetch("https://api.bybit.com/v5/order/create", "POST")


def test_public_client_has_no_keys() -> None:
    client = make_public_client("bybit")
    assert not client.apiKey and not client.secret
