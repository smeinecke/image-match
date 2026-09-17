import hashlib
import os
from contextlib import suppress
from time import sleep

import pytest

pytest.importorskip("elasticsearch", reason="elasticsearch not installed (install the 'elasticsearch' extra)")

from elasticsearch import ConnectionError, Elasticsearch, NotFoundError, RequestError

from image_match.elasticsearch_driver import SignatureES

pytestmark = pytest.mark.integration

test_img_url1 = "https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg"

INDEX_NAME = f"test_environment_{hashlib.md5(os.urandom(128)).hexdigest()[:12]}"
MAPPINGS = {
    "mappings": {"properties": {"path": {"type": "keyword"}, "metadata": {"properties": {"tenant_id": {"type": "keyword"}, "project_id": {"type": "keyword"}}}}}
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
        with suppress(NotFoundError):
            es.indices.delete(index=index_name)

    request.addfinalizer(fin)


@pytest.fixture(scope="function", autouse=True)
def cleanup_index(request, es, index_name):
    def fin():
        with suppress(NotFoundError):
            es.indices.delete(index=index_name)

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

    pytest.fail(f"Elasticsearch not running (failed to connect after {i!s} tries)")


def test_lookup_with_filter_by_metadata(ses):

    ses.add_image("test1.jpg", metadata=_metadata("foo", "project-x"), refresh_after=True)
    ses.add_image("test2.jpg", metadata=_metadata("foo", "project-x"), refresh_after=True)
    ses.add_image("test3.jpg", img="test1.jpg", metadata=_metadata("foo", "project-y"), refresh_after=True)

    ses.add_image("test2.jpg", metadata=_metadata("bar", "project-x"), refresh_after=True)

    r = ses.search_image("test1.jpg", pre_filter=_nested_filter("foo", "project-x"))
    assert len(r) == 2

    r = ses.search_image("test1.jpg", pre_filter=_nested_filter("foo", "project-z"))
    assert len(r) == 0

    r = ses.search_image("test1.jpg", pre_filter=_nested_filter("bar", "project-x"))
    assert len(r) == 1

    r = ses.search_image("test1.jpg", pre_filter=_nested_filter("bar-2", "project-x"))
    assert len(r) == 0

    r = ses.search_image("test1.jpg", pre_filter=_nested_filter("bar", "project-z"))
    assert len(r) == 0


def _metadata(tenant_id, project_id):
    return {"tenant_id": tenant_id, "project_id": project_id}


def _nested_filter(tenant_id, project_id):
    return [{"term": {"metadata.tenant_id": tenant_id}}, {"term": {"metadata.project_id": project_id}}]
