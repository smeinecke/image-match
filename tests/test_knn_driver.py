"""Integration tests for the OpenSearch k-NN drivers.

OpenSearch only — Elasticsearch 7.x has no knn_vector field type.
The index is created via knn_index_body, producing a hybrid index that the
word driver can also query (verified by test_word_driver_on_hybrid_index).
"""

import asyncio
from contextlib import suppress

import pytest

from .helpers import MAPPINGS, await_until_ready, make_async_backend, make_backend, random_index_name, wait_until_ready

pytestmark = pytest.mark.integration

DIMENSION = 648  # signature length for the default n_grid=9


def _knn_mappings() -> dict:
    """Hybrid index body: knn_vector signature + the standard test mappings."""
    from image_match.opensearch_knn_driver import knn_index_body

    body = knn_index_body(DIMENSION)
    body["mappings"]["properties"].update(MAPPINGS["mappings"]["properties"])
    return body


@pytest.fixture
def os_backend():
    return make_backend("opensearch")


def test_backend_running(os_backend):
    wait_until_ready(os_backend.client, os_backend.name)


@pytest.fixture
def sknn(os_backend, requires_download):
    """A fresh k-NN index + sync driver per test."""
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    name = random_index_name("test_knn")
    os_backend.client.indices.create(index=name, body=_knn_mappings())
    yield SignatureOpenSearchKNN(os_backend.client, index=name)
    with suppress(os_backend.exc.NotFoundError):
        os_backend.client.indices.delete(index=name)


@pytest.fixture
async def asknn(requires_download):
    """A fresh k-NN index + async driver per test."""
    from image_match.opensearch_knn_async_driver import AsyncSignatureOpenSearchKNN

    abackend = make_async_backend("opensearch")
    name = random_index_name("test_knn_async")
    await abackend.client.indices.create(index=name, body=_knn_mappings())
    yield AsyncSignatureOpenSearchKNN(abackend.client, index=name)
    with suppress(abackend.exc.NotFoundError):
        await abackend.client.indices.delete(index=name)
    await abackend.client.close()


def test_add_and_lookup(sknn):
    sknn.add_image("test1.jpg", refresh_after=True)
    r = sknn.search_image("test1.jpg")
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] == 0.0
    assert "score" in r[0]
    assert "id" in r[0]


def test_distance_consistency(sknn):
    sknn.add_image("test1.jpg")
    sknn.add_image("test2.jpg", refresh_after=True)
    r = sknn.search_image("test1.jpg")
    assert r[0]["dist"] == 0.0
    assert r[-1]["dist"] == 0.42412912927363733


def test_metadata_filter(sknn):
    metadata = {"tenant_id": "foo"}
    sknn.add_image("test1.jpg", metadata=metadata, refresh_after=True)
    sknn.add_image("test2.jpg", metadata={"tenant_id": "bar-2"}, refresh_after=True)

    r = sknn.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "foo"}})
    assert len(r) == 1
    assert r[0]["metadata"] == metadata

    r = sknn.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "bar-3"}})
    assert len(r) == 0


def test_all_orientations(sknn):
    from PIL import Image

    im = Image.open("test1.jpg")
    im.rotate(90, expand=True).save("rotated_test1.jpg")

    sknn.add_image("test1.jpg", refresh_after=True)
    r = sknn.search_image("rotated_test1.jpg", all_orientations=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] < 0.1  # some error from rotation


def test_duplicate_removal(sknn):
    for i in range(5):
        sknn.add_image("test1.jpg", refresh_after=(i == 4))
    r = sknn.search_image("test1.jpg")
    assert len(r) == 5
    sknn.delete_duplicates("test1.jpg")
    import time

    time.sleep(1)
    r = sknn.search_image("test1.jpg")
    assert len(r) == 1


def test_word_driver_on_hybrid_index(sknn, os_backend):
    """The classic word driver must still work on a knn-mapped index."""
    from image_match.opensearch_driver import SignatureOpenSearch

    sknn.add_image("test1.jpg", refresh_after=True)
    word_driver = SignatureOpenSearch(os_backend.client, index=sknn.index)
    r = word_driver.search_image("test1.jpg")
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"


async def test_async_add_and_lookup(asknn):
    await asknn.add_image("test1.jpg", refresh_after=True)
    r = await asknn.search_image("test1.jpg")
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] == 0.0


async def test_async_all_orientations(asknn):
    from PIL import Image

    im = Image.open("test1.jpg")
    im.rotate(90, expand=True).save("rotated_test1.jpg")

    await asknn.add_image("test1.jpg", refresh_after=True)
    r = await asknn.search_image("rotated_test1.jpg", all_orientations=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"


async def test_async_concurrent_searches(asknn):
    await asknn.add_image("test1.jpg")
    await asknn.add_image("test_diff.jpg", refresh_after=True)
    r1, r2 = await asyncio.gather(asknn.search_image("test1.jpg"), asknn.search_image("test_diff.jpg"))
    assert r1[0]["path"] == "test1.jpg"
    assert r2[0]["path"] == "test_diff.jpg"
