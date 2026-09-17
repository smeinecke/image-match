from contextlib import suppress

import pytest

from .helpers import MAPPINGS_NESTED, random_index_name

pytestmark = pytest.mark.integration


@pytest.fixture
def index_name(backend):
    name = random_index_name("test_nested")
    backend.client.indices.create(index=name, body=MAPPINGS_NESTED)
    yield name
    with suppress(backend.exc.NotFoundError):
        backend.client.indices.delete(index=name)


@pytest.fixture
def ses(backend, index_name):
    return backend.driver(backend.client, index=index_name)


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
