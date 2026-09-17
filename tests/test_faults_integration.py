"""Network-level fault injection via Toxiproxy.

Exercises real transport faults (connection refusal, latency beyond client
timeouts, connection resets) between the drivers and the live backends --
things mocks cannot produce. Requires the ``toxiproxy`` compose service
(``make db-up``); each test creates its proxy on demand through the
Toxiproxy HTTP API and removes it afterwards.
"""

import json
import os
import uuid
from contextlib import suppress
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

from .helpers import await_until_ready, random_index_name, wait_until_ready

pytestmark = pytest.mark.integration

TOXIPROXY_API = os.environ.get("TOXIPROXY_API", "http://localhost:8475")
PROXY_PORT = int(os.environ.get("TOXIPROXY_PROXY_PORT", "22301"))


def _toxi(method: str, path: str, payload: dict | None = None) -> dict:
    req = Request(
        f"{TOXIPROXY_API}{path}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=10) as resp:
        body = resp.read()
        return json.loads(body) if body else {}


def _toxiproxy_running() -> bool:
    try:
        _toxi("GET", "/version")
    except (URLError, OSError):
        return False
    return True


@pytest.fixture()
def proxy():
    """Toxiproxy -> opensearch proxy. Yields (name, base_url); tears the proxy down."""
    if not _toxiproxy_running():
        pytest.skip("toxiproxy not running (docker compose up -d toxiproxy)")
    name = f"test-os-{uuid.uuid4().hex[:8]}"
    _toxi(
        "POST",
        "/proxies",
        {"name": name, "listen": f"0.0.0.0:{PROXY_PORT}", "upstream": "opensearch:9200", "enabled": True},
    )
    yield name, f"http://localhost:{PROXY_PORT}"
    with suppress(Exception):
        _toxi("DELETE", f"/proxies/{name}")


def _make_driver(proxy_url: str, index: str, query_timeout: str = "10s"):
    module = pytest.importorskip("opensearchpy", reason="opensearch-py not installed")
    from image_match.opensearch_driver import SignatureOpenSearch

    # the driver maps `timeout` to a per-request HTTP timeout, which overrides
    # any socket-level timeout configured on the client itself
    return module, SignatureOpenSearch(module.OpenSearch(proxy_url), index=index, timeout=query_timeout)


def test_search_through_proxy(proxy, requires_download):
    """Sanity: a healthy proxy is transparent to add/search."""
    _, url = proxy
    module, ses = _make_driver(url, random_index_name())
    wait_until_ready(module.OpenSearch(url), "toxiproxy->opensearch")
    ses.add_image("test1.jpg", refresh_after=True)
    hits = ses.search_image("test1.jpg")
    assert len(hits) == 1
    assert hits[0]["path"] == "test1.jpg"


def test_search_fails_when_proxy_disabled(proxy, requires_download):
    """Cutting the proxy must surface as a client ConnectionError, not a hang."""
    name, url = proxy
    module, ses = _make_driver(url, random_index_name())
    wait_until_ready(module.OpenSearch(url), "toxiproxy->opensearch")
    _toxi("POST", f"/proxies/{name}", {"enabled": False})
    with pytest.raises(module.ConnectionError):
        ses.search_image("test1.jpg")


def test_search_times_out_under_latency_toxic(proxy, requires_download):
    """A latency toxic larger than the query timeout must raise ConnectionTimeout."""
    name, url = proxy
    module, ses = _make_driver(url, random_index_name(), query_timeout="1s")
    wait_until_ready(module.OpenSearch(url), "toxiproxy->opensearch")
    _toxi(
        "POST",
        f"/proxies/{name}/toxics",
        {"name": "latency", "type": "latency", "attributes": {"latency": 5000}},
    )
    with pytest.raises(module.ConnectionTimeout):
        ses.search_image("test1.jpg")


def test_search_connection_reset(proxy, requires_download):
    """reset_peer tears down the TCP connection mid-request; error must propagate."""
    name, url = proxy
    module, ses = _make_driver(url, random_index_name())
    wait_until_ready(module.OpenSearch(url), "toxiproxy->opensearch")
    _toxi(
        "POST",
        f"/proxies/{name}/toxics",
        {"name": "reset", "type": "reset_peer", "attributes": {"timeout": 0}},
    )
    with pytest.raises(module.TransportError):
        ses.search_image("test1.jpg")


async def test_async_search_fails_when_proxy_disabled(proxy, requires_download):
    """The async driver's asyncio.gather must propagate a severed connection."""
    name, url = proxy
    module = pytest.importorskip("opensearchpy", reason="opensearch-py not installed")
    if not hasattr(module, "AsyncOpenSearch"):
        pytest.skip("opensearch-py[async] not installed")
    from image_match.opensearch_async_driver import AsyncSignatureOpenSearch

    client = module.AsyncOpenSearch(url)
    try:
        await await_until_ready(client, "toxiproxy->opensearch")
        ses = AsyncSignatureOpenSearch(client, index=random_index_name())
        _toxi("POST", f"/proxies/{name}", {"enabled": False})
        with pytest.raises(module.ConnectionError):
            await ses.search_image("test1.jpg", all_orientations=True)
    finally:
        await client.close()
