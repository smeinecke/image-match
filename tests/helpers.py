"""Shared constants and factories for the backend integration tests."""

import asyncio
import os
import uuid
from contextlib import suppress
from pathlib import Path
from time import sleep
from types import SimpleNamespace
from urllib.request import urlretrieve

import pytest

# the only original test image URL that still resolves; the reference
# images themselves are committed under docs/source/_images
TEST_IMG_URL = "https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg"

DOCS_IMAGES = Path(__file__).parent.parent / "docs" / "source" / "_images"

MAPPINGS = {
    "mappings": {
        "properties": {
            "path": {"type": "keyword"},
            "metadata": {"properties": {"tenant_id": {"type": "keyword"}}},
        }
    }
}

MAPPINGS_NESTED = {
    "mappings": {
        "properties": {
            "path": {"type": "keyword"},
            "metadata": {"properties": {"tenant_id": {"type": "keyword"}, "project_id": {"type": "keyword"}}},
        }
    }
}

BACKENDS = ["elasticsearch", "opensearch", "opensearch3"]

OPENSEARCH_URLS = {
    "opensearch": ("OPENSEARCH_URL", "http://localhost:9201"),
    "opensearch3": ("OPENSEARCH3_URL", "http://localhost:9202"),
}


def random_index_name(prefix: str = "test_environment") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_backend(name: str) -> SimpleNamespace:
    """Instantiate the client + driver for a search backend.

    Skips the calling test if the backend's optional extra is not installed.
    """
    if name in OPENSEARCH_URLS:
        module = pytest.importorskip("opensearchpy", reason="opensearch-py not installed (install the 'opensearch' extra)")
        from image_match.opensearch_driver import SignatureOpenSearch

        env_var, default = OPENSEARCH_URLS[name]
        return SimpleNamespace(
            name=name,
            client=module.OpenSearch(os.environ.get(env_var, default)),
            driver=SignatureOpenSearch,
            exc=module,
        )
    module = pytest.importorskip("elasticsearch", reason="elasticsearch not installed (install the 'elasticsearch' extra)")
    from image_match.elasticsearch_driver import SignatureES

    return SimpleNamespace(
        name=name,
        client=module.Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")),
        driver=SignatureES,
        exc=module,
    )


def make_async_backend(name: str) -> SimpleNamespace:
    """Instantiate the async client + driver for a search backend.

    Skips the calling test if the backend's '-async' extra is not installed.
    """
    if name in OPENSEARCH_URLS:
        module = pytest.importorskip("opensearchpy", reason="opensearch-py not installed (install the 'opensearch-async' extra)")
        if not hasattr(module, "AsyncOpenSearch"):
            pytest.skip("opensearch-py[async] not installed (install the 'opensearch-async' extra)")
        from image_match.opensearch_async_driver import AsyncSignatureOpenSearch

        env_var, default = OPENSEARCH_URLS[name]
        return SimpleNamespace(
            name=name,
            client=module.AsyncOpenSearch(os.environ.get(env_var, default)),
            driver=AsyncSignatureOpenSearch,
            exc=module,
        )
    module = pytest.importorskip("elasticsearch", reason="elasticsearch not installed (install the 'elasticsearch-async' extra)")
    if not hasattr(module, "AsyncElasticsearch"):
        pytest.skip("elasticsearch[async] not installed (install the 'elasticsearch-async' extra)")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    return SimpleNamespace(
        name=name,
        client=module.AsyncElasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")),
        driver=AsyncSignatureES,
        exc=module,
    )


def wait_until_ready(client, name: str) -> None:
    """Ping a backend a few times before giving up."""
    for _ in range(5):
        with suppress(Exception):
            if client.ping():
                return
        sleep(2)
    pytest.fail(f"{name} not running (failed to connect)")


async def await_until_ready(client, name: str) -> None:
    """Async variant of wait_until_ready for async clients."""
    for _ in range(5):
        with suppress(Exception):
            if await client.ping():
                return
        await asyncio.sleep(2)
    pytest.fail(f"{name} not running (failed to connect)")


def download(url: str, dest: Path) -> None:
    """Download a test image; offline runs leave the file absent."""
    with suppress(OSError):
        urlretrieve(url, dest)
