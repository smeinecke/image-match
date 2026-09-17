"""Unit tests for the database drivers using mocked clients — no services needed."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from image_match.signature_database_base import (
    SignatureDatabaseBase,
    dedupe_results,
    get_words,
    make_record,
    max_contrast,
    normalized_distance,
    words_to_int,
)


def _rec(path="img.jpg", n_words=5):
    """A minimal make_record-shaped dict with a random signature."""
    rec = {"path": path, "signature": np.random.default_rng(0).integers(-2, 3, size=64).tolist()}
    for i in range(n_words):
        rec[f"simple_word_{i}"] = i * 100
    return rec


# --- signature_database_base helpers --------------------------------------


def test_make_record_structure():
    from image_match.goldberg import ImageSignature

    rec = make_record("test.jpg", ImageSignature(), k=16, N=63, metadata={"a": 1})
    assert rec["path"] == "test.jpg"
    assert len(rec["signature"]) == 648
    assert rec["metadata"] == {"a": 1}
    assert sum(1 for k in rec if k.startswith("simple_word_")) == 63


def test_make_record_no_metadata():
    from image_match.goldberg import ImageSignature

    rec = make_record("test.jpg", ImageSignature(), k=16, N=63)
    assert "metadata" not in rec


def test_get_words_overlapping():
    words = get_words(np.arange(10, dtype=np.int8), k=4, N=5)
    assert words.shape == (5, 4)


def test_get_words_validation():
    with pytest.raises(ValueError, match="Word length"):
        get_words(np.arange(4), k=10, N=2)
    with pytest.raises(ValueError, match="Number of words"):
        get_words(np.arange(4), k=1, N=10)


def test_words_to_int_encoding():
    # [-1,-1,-1] -> 0 ; [0,0,0] -> 13 ; [0,1,0] -> 16  (docstring examples)
    out = words_to_int(np.array([[-1, -1, -1], [0, 0, 0], [0, 1, 0]]))
    assert out.tolist() == [0, 13, 16]


def test_max_contrast():
    arr = np.array([[-2, 0, 5], [1, -1, 0]], dtype=np.int8)
    max_contrast(arr)
    assert arr.tolist() == [[-1, 0, 1], [1, -1, 0]]


def test_normalized_distance_zero_and_nan():
    a = np.array([[1.0, 2.0, 3.0]])
    assert normalized_distance(a, np.array([1.0, 2.0, 3.0]))[0] == 0.0
    # zero vector vs zero vector -> 0/0 -> nan_value
    assert normalized_distance(np.zeros((1, 3)), np.zeros(3), nan_value=0.99)[0] == 0.99


def test_dedupe_results():
    res = [
        {"id": "a", "dist": 0.5},
        {"id": "b", "dist": 0.1},
        {"id": "a", "dist": 0.4},  # dup id dropped
    ]
    out = dedupe_results(res)
    assert [r["id"] for r in out] == ["b", "a"]
    assert out[0]["dist"] < out[1]["dist"]


def test_base_init_validation():
    with pytest.raises(TypeError):
        SignatureDatabaseBase(k="x")
    with pytest.raises(TypeError):
        SignatureDatabaseBase(N=1.5)
    with pytest.raises(TypeError):
        SignatureDatabaseBase(n_grid=None)
    with pytest.raises(TypeError):
        SignatureDatabaseBase(distance_cutoff="hi")
    with pytest.raises(ValueError):
        SignatureDatabaseBase(distance_cutoff=-1)


def test_base_abstract_methods():
    base = SignatureDatabaseBase()
    with pytest.raises(NotImplementedError):
        base.search_single_record({})
    with pytest.raises(NotImplementedError):
        base.insert_single_record({})


class _StubDriver(SignatureDatabaseBase):
    """Minimal concrete driver for exercising search_image/add_image logic."""

    def __init__(self):
        super().__init__()
        self.inserted = []
        self.results = []

    def search_single_record(self, rec, pre_filter=None, **kwargs):
        return list(self.results)

    def insert_single_record(self, rec, **kwargs):
        self.inserted.append(rec)


def test_add_image_delegates():
    d = _StubDriver()
    d.add_image("test.jpg")
    assert len(d.inserted) == 1
    assert d.inserted[0]["path"] == "test.jpg"


def test_search_image_dedupes_and_sorts():
    d = _StubDriver()
    d.results = [{"id": "x", "dist": 0.2}, {"id": "x", "dist": 0.2}]
    out = d.search_image("test.jpg")
    assert len(out) == 1


def test_search_image_all_orientations_queries_each():
    d = _StubDriver()
    seen = []
    d.search_single_record = lambda rec, pre_filter=None, **kw: seen.append(rec) or []
    d.search_image("test.jpg", all_orientations=True)
    # 2 inversions x 4 rotations x 2 mirrors = 16 records
    assert len(seen) == 16


# --- Elasticsearch driver --------------------------------------------------


def _es_hits(n=1, signature=None, path="img.jpg"):
    sig = signature if signature is not None else np.zeros(64).tolist()
    return {"hits": {"hits": [{"_id": f"id{i}", "_score": 1.0, "_source": {"path": path, "signature": sig}} for i in range(n)]}}


def test_es_search_single_record_query():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = _es_hits()
    ses = SignatureES(es, index="idx", size=7)
    rec = _rec()
    rec["timestamp"] = "stray"  # stored-field must not leak into query
    rec["metadata"] = {"t": "x"}
    ses.search_single_record(rec)

    call = es.search.call_args
    assert call.kwargs["index"] == "idx"
    assert call.kwargs["size"] == 7
    body = call.kwargs["body"]
    should = body["query"]["bool"]["should"]
    # only simple_word_* terms, no timestamp/metadata/path
    assert len(should) == 5
    assert all("simple_word_" in next(iter(t["term"])) for t in should)


def test_es_search_single_record_pre_filter():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = _es_hits()
    ses = SignatureES(es)
    ses.search_single_record(_rec(), pre_filter={"term": {"metadata.tenant_id": "foo"}})
    body = es.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == {"term": {"metadata.tenant_id": "foo"}}


def test_es_search_single_record_empty():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {"hits": []}}
    assert SignatureES(es).search_single_record(_rec()) == []


def test_es_search_single_record_cutoff():
    from image_match.elasticsearch_driver import SignatureES

    sig = np.ones(64).tolist()
    es = MagicMock()
    es.search.return_value = _es_hits(n=1, signature=sig)
    ses = SignatureES(es, distance_cutoff=0.45)
    rec = _rec()
    rec["signature"] = list(sig)
    r = ses.search_single_record(rec)
    assert len(r) == 1
    assert r[0]["dist"] == 0.0

    ses.distance_cutoff = -0.5  # nothing below this
    rec2 = _rec()
    rec2["signature"] = list(sig)
    assert ses.search_single_record(rec2) == []


def test_es_insert_single_record():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    ses = SignatureES(es)
    rec = _rec()
    ses.insert_single_record(rec, refresh_after=True)
    call = es.index.call_args
    assert call.kwargs["index"] == "images"
    assert call.kwargs["document"] is rec
    assert call.kwargs["refresh"] is True
    assert "timestamp" in rec


def test_es_delete_duplicates():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "keep", "_source": {"path": "p"}},
                {"_id": "del1", "_source": {"path": "p"}},
                {"_id": "other", "_source": {"path": "different"}},  # fuzzy match, not exact
                {"_id": "nopath", "_source": {}},  # legacy doc without path
                {"_id": "del2", "_source": {"path": "p"}},
            ]
        }
    }
    SignatureES(es).delete_duplicates("p")
    deleted = sorted(c.kwargs["id"] for c in es.delete.call_args_list)
    assert deleted == ["del1", "del2"]


def test_es_delete_duplicates_limit():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {"hits": []}}
    ses = SignatureES(es, delete_duplicates_limit=42)
    ses.delete_duplicates("p")
    assert es.search.call_args.kwargs["size"] == 42
    ses.delete_duplicates("p", limit=7)
    assert es.search.call_args.kwargs["size"] == 7


# --- OpenSearch driver -----------------------------------------------------


def test_opensearch_search_uses_params():
    from image_match.opensearch_driver import SignatureOpenSearch, _parse_duration

    client = MagicMock()
    client.search.return_value = {"hits": {"hits": []}}
    ses = SignatureOpenSearch(client, index="idx", size=55, timeout="500ms")
    ses.search_single_record(_rec())

    call = client.search.call_args
    assert call.kwargs["params"]["size"] == 55
    assert call.kwargs["params"]["request_timeout"] == 0.5
    assert "size" not in call.kwargs  # reserved kwargs must not be sent

    assert _parse_duration("10s") == 10.0
    assert _parse_duration("1m") == 60.0
    assert _parse_duration("250") == 250.0


def test_opensearch_insert_body_and_refresh():
    from image_match.opensearch_driver import SignatureOpenSearch

    client = MagicMock()
    ses = SignatureOpenSearch(client)
    rec = _rec()
    ses.insert_single_record(rec, refresh_after=True)
    call = client.index.call_args
    assert call.kwargs["body"] is rec
    assert call.kwargs["params"]["refresh"] == "true"
    assert "timestamp" in rec


# --- MongoDB driver --------------------------------------------------------


def _mongo_collection(docs=None, n_indexes=1):
    coll = MagicMock()
    coll.count_documents.return_value = len(docs or [])
    coll.find_one.return_value = (docs or [None])[0]
    coll.index_information.return_value = {f"idx_{i}": {} for i in range(n_indexes)}
    return coll


def test_mongo_init_empty_collection():
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[])
    ses = SignatureMongo(coll)
    assert ses.index_names == []


def test_mongo_init_loads_index_names():
    from image_match.mongodb_driver import SignatureMongo

    doc = {"path": "x", "signature": [], "simple_word_0": 1, "simple_word_1": 2}
    ses = SignatureMongo(_mongo_collection(docs=[doc]))
    assert ses.index_names == ["simple_word_0", "simple_word_1"]


def test_mongo_init_find_one_none():
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection()
    coll.count_documents.return_value = 5
    coll.find_one.return_value = None  # raced delete
    ses = SignatureMongo(coll)
    assert ses.index_names == []


def test_mongo_insert_creates_indexes_once():
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 1, "simple_word_9": 2}])
    ses = SignatureMongo(coll)
    ses.index_names = ["simple_word_0", "simple_word_9"]
    ses.insert_single_record(_rec())
    coll.insert_one.assert_called_once()
    # only _id index existed -> indexes created
    assert coll.create_index.call_count == 2

    coll.reset_mock()
    coll.index_information.return_value = {"_id_": {}, "w0": {}, "w9": {}}
    ses.insert_single_record(_rec())
    coll.create_index.assert_not_called()  # already indexed


def test_mongo_word_query_queue():
    from image_match.mongodb_driver import SignatureMongo

    ses = SignatureMongo(_mongo_collection())
    # index built with more words than the record has -> absent field skipped
    ses.index_names = ["simple_word_0", "simple_word_1", "simple_word_2", "simple_word_7"]
    rec = _rec()  # only has simple_word_0..4

    q = ses._word_query_queue(rec, word_limit=4)
    items = []
    while True:
        it = q.get()
        if it == "STOP":
            break
        items.append(it)
    assert items == [{"simple_word_0": 0}, {"simple_word_1": 100}, {"simple_word_2": 200}]


def test_mongo_search_single_record():
    from image_match.mongodb_driver import SignatureMongo

    # non-zero signature: all-zero vectors produce 0/0 -> nan -> max distance
    target_sig = np.arange(64) % 3 - 1
    doc = {"_id": "m1", "signature": target_sig.tolist(), "path": "p.jpg"}
    coll = _mongo_collection(docs=[{"simple_word_0": 0}])
    coll.find.return_value = iter([doc])
    ses = SignatureMongo(coll)
    ses.index_names = ["simple_word_0"]
    rec = _rec()
    rec["signature"] = target_sig.tolist()
    r = ses.search_single_record(rec)
    assert len(r) == 1
    assert r[0]["id"] == "m1"
    assert r[0]["dist"] == 0.0


def test_mongo_search_n_parallel_words_validation():
    from image_match.mongodb_driver import SignatureMongo

    ses = SignatureMongo(_mongo_collection())
    with pytest.raises(ValueError, match="n_parallel_words"):
        ses.search_single_record(_rec(), n_parallel_words=0)


def test_mongo_search_skips_popular_words():
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 0}])
    coll.count_documents.side_effect = lambda q: 99999  # over maximum_matches
    ses = SignatureMongo(coll)
    ses.index_names = ["simple_word_0"]
    assert ses.search_single_record(_rec(), maximum_matches=100) == []
    coll.find.assert_not_called()


def test_get_next_match_pre_filter():
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    coll.count_documents.return_value = 1
    coll.find.return_value = iter([])
    get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64), pre_filter={"metadata.t": "x"})
    # query merged filter + word
    assert coll.find.call_args[0][0] == {"simple_word_0": 5, "metadata.t": "x"}
    assert rq.get() == "STOP"  # sentinel always enqueued


# --- Async drivers ---------------------------------------------------------


def _async_es():
    es = MagicMock()
    es.search = AsyncMock(return_value=_es_hits())
    es.index = AsyncMock()
    es.delete = AsyncMock()
    return es


async def test_async_es_search_single_record():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    ses = AsyncSignatureES(es, index="idx")
    rec = _rec()
    r = await ses.search_single_record(rec)
    es.search.assert_awaited_once()
    assert isinstance(r, list)


async def test_async_es_insert_and_delete_duplicates():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    ses = AsyncSignatureES(es)
    await ses.insert_single_record(_rec(), refresh_after=True)
    assert es.index.await_args.kwargs["refresh"] is True

    es.search = AsyncMock(return_value={"hits": {"hits": [{"_id": "k", "_source": {"path": "p"}}, {"_id": "d", "_source": {"path": "p"}}]}})
    await ses.delete_duplicates("p")
    es.delete.assert_awaited_once()
    assert es.delete.await_args.kwargs["id"] == "d"


async def test_async_es_add_and_search_image():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    ses = AsyncSignatureES(es)
    await ses.add_image("test.jpg", refresh_after=True)
    es.index.assert_awaited_once()

    # return a hit whose signature equals the freshly generated one -> dist 0
    rec = make_record("test.jpg", ses.gis, ses.k, ses.N)
    es.search = AsyncMock(return_value=_es_hits(n=1, signature=rec["signature"]))
    r = await ses.search_image("test.jpg")
    assert len(r) == 1
    assert r[0]["dist"] == 0.0


async def test_async_opensearch_params():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.opensearch_async_driver import AsyncSignatureOpenSearch

    client = _async_es()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    ses = AsyncSignatureOpenSearch(client, size=33, timeout="1m")
    await ses.search_single_record(_rec())
    call = client.search.await_args
    assert call.kwargs["params"]["size"] == 33
    assert call.kwargs["params"]["request_timeout"] == 60.0

    rec = _rec()
    await ses.insert_single_record(rec, refresh_after=True)
    call = client.index.await_args
    assert call.kwargs["body"] is rec
    assert call.kwargs["params"]["refresh"] == "true"
