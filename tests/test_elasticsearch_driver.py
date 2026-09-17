import hashlib
import os
import shutil
from time import sleep
from urllib.request import urlretrieve

import pytest
from elasticsearch import ConnectionError, Elasticsearch, NotFoundError, RequestError
from PIL import Image

from image_match.elasticsearch_driver import SignatureES

pytestmark = pytest.mark.integration

# the original test URLs are no longer reachable; the flickr URL still
# resolves, and docs/source/_images holds copies of the reference images
DOCS_IMAGES = os.path.join(os.path.dirname(__file__), "..", "docs", "source", "_images")
test_img_url1 = "https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg"
test_img_url2 = test_img_url1
try:
    # import-time download; tests that need the file fail clearly if offline
    urlretrieve(test_img_url1, "test1.jpg")
except OSError:
    pass
shutil.copy(os.path.join(DOCS_IMAGES, "MonaLisa_Wikipedia.jpg"), "test2.jpg")

INDEX_NAME = "test_environment_{}".format(hashlib.md5(os.urandom(128)).hexdigest()[:12])
MAPPINGS = {
    "mappings": {
        "properties": {
            "path": {"type": "keyword"},
            "metadata": {
                "properties": {
                    "tenant_id": {
                        "type": "keyword",
                    }
                }
            },
        }
    }
}


@pytest.fixture(scope="module", autouse=True)
def index_name():
    return INDEX_NAME


@pytest.fixture(scope="function", autouse=True)
def setup_index(request, index_name):
    es = Elasticsearch()
    try:
        es.indices.create(index=index_name, body=MAPPINGS)
    except RequestError as e:
        if e.error == "resource_already_exists_exception":
            es.indices.delete(index=index_name)
        else:
            raise

    def fin():
        try:
            es.indices.delete(index=index_name)
        except NotFoundError:
            pass

    request.addfinalizer(fin)


@pytest.fixture(scope="function", autouse=True)
def cleanup_index(request, es, index_name):
    def fin():
        try:
            es.indices.delete(index=index_name)
        except NotFoundError:
            pass

    request.addfinalizer(fin)


@pytest.fixture
def es():
    return Elasticsearch()


@pytest.fixture
def ses(es, index_name):
    return SignatureES(es=es, index=index_name)


def test_elasticsearch_running(es):
    i = 0
    while i < 5:
        try:
            es.ping()
            assert True
            return
        except ConnectionError:
            i += 1
            sleep(2)

    pytest.fail("Elasticsearch not running (failed to connect after {} tries)".format(str(i)))


def test_add_image_by_url(ses):
    ses.add_image(test_img_url1)
    ses.add_image(test_img_url2)
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
    r = ses.search_image(test_img_url1)
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


def check_distance_consistency(ses):
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
    metadata = dict(tenant_id="foo")
    ses.add_image("test1.jpg", metadata=metadata, refresh_after=True)

    metadata2 = dict(tenant_id="bar-2")
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
        ses.add_image("test1.jpg")
    sleep(1)
    r = ses.search_image("test1.jpg")
    assert len(r) == 10
    ses.delete_duplicates("test1.jpg")
    sleep(1)
    r = ses.search_image("test1.jpg")
    assert len(r) == 1
