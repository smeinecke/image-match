"""Integration tests for tools/migrate_to_knn.py.

Covers Elasticsearch -> OpenSearch and OpenSearch -> OpenSearch migration
into a hybrid k-NN index, then verifies both the classic word driver and the
new knn driver can search the migrated index.
"""

import importlib.util
from contextlib import suppress
from pathlib import Path

import pytest

from .helpers import MAPPINGS, make_backend, random_index_name, wait_until_ready

pytestmark = pytest.mark.integration

DIMENSION = 648


def _load_tool():
    spec = importlib.util.spec_from_file_location("migrate_to_knn", Path(__file__).parent.parent / "tools" / "migrate_to_knn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


migrate_tool = _load_tool()


@pytest.fixture
def os_backend():
    return make_backend("opensearch")


@pytest.fixture
def es_backend():
    return make_backend("elasticsearch")


def _seed_word_index(backend, images=("test1.jpg", "test2.jpg")):
    """Create a word-overlap index on `backend` with `images` indexed."""
    name = random_index_name("test_mig_src")
    backend.client.indices.create(index=name, body=MAPPINGS)
    driver = backend.driver(backend.client, index=name)
    for img in images:
        driver.add_image(img, refresh_after=True)
    return name


def _teardown(client, name, exc):
    with suppress(exc.NotFoundError):
        client.indices.delete(index=name)


def test_detect_server_type(es_backend, os_backend):
    wait_until_ready(es_backend.client, "elasticsearch")
    wait_until_ready(os_backend.client, "opensearch")
    assert migrate_tool.detect_server_type(es_backend.client) == "elasticsearch"
    assert migrate_tool.detect_server_type(os_backend.client) == "opensearch"


def test_migrate_es_to_os_knn(es_backend, os_backend, requires_download):
    """Full ES -> OS k-NN migration: old and new drivers both work after."""
    wait_until_ready(es_backend.client, "elasticsearch")
    wait_until_ready(os_backend.client, "opensearch")

    src = _seed_word_index(es_backend)
    dst = random_index_name("test_mig_dst")

    try:
        stats = migrate_tool.migrate_index(es_backend.client, os_backend.client, src, dst, dimension=DIMENSION)
        assert stats["scanned"] == 2
        assert stats["indexed"] == 2
        assert stats["skipped"] == 0
        assert migrate_tool.verify_index(os_backend.client, dst, DIMENSION)

        os_backend.client.indices.refresh(index=dst)

        # the new knn driver finds the golden match on the migrated index
        from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

        knn_driver = SignatureOpenSearchKNN(os_backend.client, index=dst)
        r = knn_driver.search_image("test1.jpg")
        assert r[0]["path"] == "test1.jpg"
        assert r[0]["dist"] == 0.0
        assert r[-1]["dist"] == 0.42412912927363733

        # and the classic word driver still works on the hybrid index (rollback)
        word_driver = os_backend.driver(os_backend.client, index=dst)
        r = word_driver.search_image("test1.jpg")
        assert r[0]["path"] == "test1.jpg"
        assert r[0]["dist"] == 0.0
    finally:
        _teardown(es_backend.client, src, es_backend.exc)
        _teardown(os_backend.client, dst, os_backend.exc)


def test_migrate_os_to_os_knn(os_backend, requires_download):
    wait_until_ready(os_backend.client, "opensearch")

    src = _seed_word_index(os_backend)
    dst = random_index_name("test_mig_dst")

    try:
        stats = migrate_tool.migrate_index(os_backend.client, os_backend.client, src, dst, dimension=DIMENSION)
        assert stats["indexed"] == 2

        os_backend.client.indices.refresh(index=dst)

        from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

        r = SignatureOpenSearchKNN(os_backend.client, index=dst).search_image("test1.jpg")
        assert r[0]["path"] == "test1.jpg"
        assert r[0]["dist"] == 0.0
    finally:
        _teardown(os_backend.client, src, os_backend.exc)
        _teardown(os_backend.client, dst, os_backend.exc)


def test_migrate_skips_bad_dimensions(os_backend, requires_download):
    """Docs whose signature doesn't match --dimension are skipped, not fatal."""
    wait_until_ready(os_backend.client, "opensearch")

    src = _seed_word_index(os_backend, images=("test1.jpg",))
    dst = random_index_name("test_mig_dst")

    try:
        stats = migrate_tool.migrate_index(os_backend.client, os_backend.client, src, dst, dimension=64)
        assert stats["scanned"] == 1
        assert stats["indexed"] == 0
        assert stats["skipped"] == 1
    finally:
        _teardown(os_backend.client, src, os_backend.exc)
        _teardown(os_backend.client, dst, os_backend.exc)


def test_migrate_rerun_is_idempotent(os_backend, requires_download):
    """Preserving _id means a second run overwrites rather than duplicates."""
    wait_until_ready(os_backend.client, "opensearch")

    src = _seed_word_index(os_backend, images=("test1.jpg",))
    dst = random_index_name("test_mig_dst")

    try:
        migrate_tool.migrate_index(os_backend.client, os_backend.client, src, dst, dimension=DIMENSION)
        stats = migrate_tool.migrate_index(os_backend.client, os_backend.client, src, dst, dimension=DIMENSION)
        assert stats["indexed"] == 1
        os_backend.client.indices.refresh(index=dst)
        assert os_backend.client.count(index=dst)["count"] == 1
    finally:
        _teardown(os_backend.client, src, os_backend.exc)
        _teardown(os_backend.client, dst, os_backend.exc)
