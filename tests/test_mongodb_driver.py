import os
import shutil
from urllib.request import urlretrieve

import pytest
from PIL import Image
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from image_match.mongodb_driver import SignatureMongo

pytestmark = pytest.mark.integration

# the original test URLs are no longer reachable; the flickr URL still
# resolves, and docs/source/_images holds copies of the reference images
DOCS_IMAGES = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_images")
test_img_url1 = "https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg"
urlretrieve(test_img_url1, "test1.jpg")
shutil.copy(os.path.join(DOCS_IMAGES, "MonaLisa_Wikipedia.jpg"), "test2.jpg")


@pytest.fixture
def collection():
    client = MongoClient(serverSelectionTimeoutMS=2000)
    coll = client.images.test_images
    yield coll
    coll.drop()
    client.close()


@pytest.fixture
def ses(collection):
    return SignatureMongo(collection)


def test_mongodb_running(collection):
    try:
        collection.database.client.admin.command("ping")
    except ServerSelectionTimeoutError:
        pytest.fail("MongoDB not running (failed to connect)")


def test_add_image_by_url(ses):
    ses.add_image(test_img_url1)
    assert True


def test_add_image_by_path(ses):
    ses.add_image("test1.jpg")
    assert True


def test_add_image_as_bytestream(ses):
    with open("test1.jpg", "rb") as f:
        ses.add_image("bytestream_test", img=f.read(), bytestream=True)
    assert True


def test_add_image_with_different_name(ses):
    ses.add_image("custom_name_test", img="test1.jpg", bytestream=False)
    assert True


def test_lookup_from_file(ses):
    ses.add_image("test1.jpg")
    r = ses.search_image("test1.jpg")
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_from_bytestream(ses):
    ses.add_image("test1.jpg")
    with open("test1.jpg", "rb") as f:
        r = ses.search_image(f.read(), bytestream=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_with_cutoff(ses):
    ses.add_image("test2.jpg")
    ses.distance_cutoff = 0.01
    r = ses.search_image("test1.jpg")
    assert len(r) == 0


def test_add_image_with_metadata(ses):
    metadata = {"some_info": {"test": "ok!"}}
    ses.add_image("test1.jpg", metadata=metadata)
    r = ses.search_image("test1.jpg")
    assert r[0]["metadata"] == metadata
    assert "path" in r[0]
    assert "dist" in r[0]
    assert "id" in r[0]


def test_lookup_with_filter_by_metadata(ses):
    metadata = dict(tenant_id="foo")
    ses.add_image("test1.jpg", metadata=metadata)

    metadata2 = dict(tenant_id="bar-2")
    ses.add_image("test2.jpg", metadata=metadata2)

    r = ses.search_image("test1.jpg", pre_filter={"metadata.tenant_id": "foo"})
    assert len(r) == 1
    assert r[0]["metadata"] == metadata

    r = ses.search_image("test1.jpg", pre_filter={"metadata.tenant_id": "bar-2"})
    assert len(r) == 1
    assert r[0]["metadata"] == metadata2

    r = ses.search_image("test1.jpg", pre_filter={"metadata.tenant_id": "bar-3"})
    assert len(r) == 0


def test_all_orientations(ses):
    im = Image.open("test1.jpg")
    im.rotate(90, expand=True).save("rotated_test1.jpg")

    ses.add_image("test1.jpg")
    r = ses.search_image("rotated_test1.jpg", all_orientations=True)
    assert len(r) == 1
    assert r[0]["path"] == "test1.jpg"
    assert r[0]["dist"] < 0.1  # some error from rotation

    with open("rotated_test1.jpg", "rb") as f:
        r = ses.search_image(f.read(), bytestream=True, all_orientations=True)
        assert len(r) == 1
        assert r[0]["dist"] < 0.1  # some error from rotation


def test_duplicate(ses):
    ses.add_image("test1.jpg")
    ses.add_image("test1.jpg")
    r = ses.search_image("test1.jpg")
    assert len(r) == 2
    assert r[0]["path"] == "test1.jpg"
    assert "dist" in r[0]
    assert "id" in r[0]


def test_distance_consistency(ses):
    ses.add_image("test1.jpg")
    ses.add_image("test2.jpg")
    r = ses.search_image("test1.jpg")
    assert len(r) == 2
    assert r[0]["dist"] == 0.0
    assert r[-1]["dist"] == 0.42412912927363733


def test_parallel_words(ses):
    ses.add_image("test1.jpg")
    ses.add_image("test2.jpg")
    r = ses.search_image("test1.jpg", n_parallel_words=4)
    assert len(r) == 2
    assert r[0]["dist"] == 0.0
