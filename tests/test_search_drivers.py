from contextlib import suppress
from time import sleep

import pytest
from PIL import Image

from .helpers import MAPPINGS, TEST_IMG_URL, random_index_name, wait_until_ready

pytestmark = pytest.mark.integration


@pytest.fixture
def index_name(backend):
    """A fresh, uniquely-named index per test."""
    name = random_index_name()
    backend.client.indices.create(index=name, body=MAPPINGS)
    yield name
    with suppress(backend.exc.NotFoundError):
        backend.client.indices.delete(index=name)


@pytest.fixture
def ses(backend, index_name, requires_download):
    return backend.driver(backend.client, index=index_name)


def test_backend_running(backend):
    wait_until_ready(backend.client, backend.name)


def test_add_image_by_url(ses):
    ses.add_image(TEST_IMG_URL)
    ses.add_image(TEST_IMG_URL)
    assert True


def test_add_image_by_path(ses):
    ses.add_image("test1.jpg")
    assert True


def test_index_refresh(ses):
    ses.add_image("test1.jpg", refresh_after=True)
    r = ses.search_image("test1.jpg")
    assert len(r) == 1


def test_add_image_as_bytestream(ses):
    with open("test1.jpg", "rb") as f:
        ses.add_image("bytestream_test", img=f.read(), bytestream=True)
    assert True


def test_add_image_with_different_name(ses):
    ses.add_image("custom_name_test", img="test1.jpg", bytestream=False)
    assert True


def test_lookup_from_url(ses):
    ses.add_image("test1.jpg", refresh_after=True)
    r = ses.search_image(TEST_IMG_URL)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert "score" in r[0]
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_from_file(ses):
    ses.add_image("test1.jpg", refresh_after=True)
    r = ses.search_image("test1.jpg")
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert "score" in r[0]
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_from_bytestream(ses):
    ses.add_image("test1.jpg", refresh_after=True)
    with open("test1.jpg", "rb") as f:
        r = ses.search_image(f.read(), bytestream=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert "score" in r[0]
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_with_cutoff(ses):
    ses.add_image("test2.jpg", refresh_after=True)
    ses.distance_cutoff = 0.01
    r = ses.search_image("test1.jpg")
    assert len(r) == 0


def test_distance_consistency(ses):
    ses.add_image("test1.jpg")
    ses.add_image("test2.jpg", refresh_after=True)
    r = ses.search_image("test1.jpg")
    assert r[0]["dist"] == 0.0
    assert r[-1]["dist"] == 0.42412912927363733


def test_add_image_with_metadata(ses):
    metadata = {"some_info": {"test": "ok!"}}
    ses.add_image("test1.jpg", metadata=metadata, refresh_after=True)
    r = ses.search_image("test1.jpg")
    assert r[0]["metadata"] == metadata
    assert "path" in r[0]
    assert "score" in r[0]
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_with_filter_by_metadata(ses):
    metadata = {"tenant_id": "foo"}
    ses.add_image("test1.jpg", metadata=metadata, refresh_after=True)

    metadata2 = {"tenant_id": "bar-2"}
    ses.add_image("test2.jpg", metadata=metadata2, refresh_after=True)

    r = ses.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "foo"}})
    assert len(r) == 1
    assert r[0]["metadata"] == metadata

    r = ses.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "bar-2"}})
    assert len(r) == 1
    assert r[0]["metadata"] == metadata2

    r = ses.search_image("test1.jpg", pre_filter={"term": {"metadata.tenant_id": "bar-3"}})
    assert len(r) == 0


def test_all_orientations(ses):
    im = Image.open("test1.jpg")
    im.rotate(90, expand=True).save("rotated_test1.jpg")

    ses.add_image("test1.jpg", refresh_after=True)
    r = ses.search_image("rotated_test1.jpg", all_orientations=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] < 0.1  # some error from rotation

    with open("rotated_test1.jpg", "rb") as f:
        r = ses.search_image(f.read(), bytestream=True, all_orientations=True)
        assert len(r) == 1
        assert r[0]["dist"] < 0.1  # some error from rotation


def test_duplicate(ses):
    ses.add_image("test1.jpg", refresh_after=True)
    ses.add_image("test1.jpg", refresh_after=True)
    r = ses.search_image("test1.jpg")
    assert len(r) == 2
    assert r[0]["path"] == "test1.jpg"
    assert "score" in r[0]
    assert "dist" in r[0]
    assert "id" in r[0]


def test_duplicate_removal(ses):
    for i in range(10):
        ses.add_image("test1.jpg", refresh_after=(i == 9))
    r = ses.search_image("test1.jpg")
    assert len(r) == 10
    ses.delete_duplicates("test1.jpg")
    sleep(1)
    r = ses.search_image("test1.jpg")
    assert len(r) == 1
