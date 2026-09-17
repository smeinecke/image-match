import asyncio
from contextlib import suppress

import pytest

from .helpers import MAPPINGS, await_until_ready, make_async_backend, random_index_name

pytestmark = pytest.mark.integration


@pytest.fixture(params=["elasticsearch", "opensearch"])
def abackend(request):
    """An async search backend (client + driver), parametrized over ES and OpenSearch."""
    return make_async_backend(request.param)


@pytest.fixture
async def ases(abackend, requires_download):
    """A fresh uniquely-named index per test; closes the async client after."""
    name = random_index_name()
    await abackend.client.indices.create(index=name, body=MAPPINGS)
    yield abackend.driver(abackend.client, index=name)
    with suppress(abackend.exc.NotFoundError):
        await abackend.client.indices.delete(index=name)
    await abackend.client.close()


async def test_backend_running(abackend):
    await await_until_ready(abackend.client, abackend.name)


async def test_add_and_lookup(ases):
    await ases.add_image("test1.jpg", refresh_after=True)
    r = await ases.search_image("test1.jpg")
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] == 0.0
    assert "score" in r[0]
    assert "id" in r[0]


async def test_lookup_from_bytestream(ases):
    await ases.add_image("test1.jpg", refresh_after=True)
    with open("test1.jpg", "rb") as f:
        r = await ases.search_image(f.read(), bytestream=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"


async def test_distance_consistency(ases):
    await ases.add_image("test1.jpg")
    await ases.add_image("test2.jpg", refresh_after=True)
    r = await ases.search_image("test1.jpg")
    assert r[0]["dist"] == 0.0
    assert r[-1]["dist"] == 0.42412912927363733


async def test_metadata_filter(ases):
    metadata = {"tenant_id": "foo"}
    await ases.add_image("test1.jpg", metadata=metadata, refresh_after=True)
    await ases.add_image("test2.jpg", metadata={"tenant_id": "bar-2"}, refresh_after=True)

    r = await ases.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "foo"}})
    assert len(r) == 1
    assert r[0]["metadata"] == metadata

    r = await ases.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "bar-3"}})
    assert len(r) == 0


async def test_concurrent_searches(ases):
    """Concurrent awaited searches return independently — the point of the driver."""
    await ases.add_image("test1.jpg")
    await ases.add_image("test_diff.jpg", refresh_after=True)
    r1, r2 = await asyncio.gather(ases.search_image("test1.jpg"), ases.search_image("test_diff.jpg"))
    assert r1[0]["path"] == "test1.jpg"
    assert r2[0]["path"] == "test_diff.jpg"


async def test_duplicate_removal(ases):
    for i in range(5):
        await ases.add_image("test1.jpg", refresh_after=(i == 4))
    r = await ases.search_image("test1.jpg")
    assert len(r) == 5
    await ases.delete_duplicates("test1.jpg")
    await asyncio.sleep(1)
    r = await ases.search_image("test1.jpg")
    assert len(r) == 1


async def test_all_orientations(ases):
    from PIL import Image

    im = Image.open("test1.jpg")
    im.rotate(90, expand=True).save("rotated_test1.jpg")

    await ases.add_image("test1.jpg", refresh_after=True)
    r = await ases.search_image("rotated_test1.jpg", all_orientations=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] < 0.1  # some error from rotation


async def test_add_image_variants(ases):
    with open("test1.jpg", "rb") as f:
        await ases.add_image("bytestream_test", img=f.read(), bytestream=True, refresh_after=True)
    await ases.add_image("custom_name_test", img="test1.jpg", bytestream=False, refresh_after=True)
    r = await ases.search_image("test1.jpg")
    paths = {x["path"] for x in r}
    assert {"bytestream_test", "custom_name_test"} <= paths


async def test_lookup_with_cutoff(ases):
    await ases.add_image("test2.jpg", refresh_after=True)
    ases.distance_cutoff = 0.01
    r = await ases.search_image("test1.jpg")
    assert len(r) == 0
