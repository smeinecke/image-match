"""Fault-injection and negative-path tests — backend failures, malformed
responses, and invalid inputs. All mocked; no services needed.

These tests document the error contracts: driver methods let backend
exceptions propagate, helpers raise KeyError on malformed records/hits, and
the mongo worker always enqueues its STOP sentinel even when the cursor
blows up.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from .test_drivers_unit import _async_es, _es_hits, _mongo_collection, _rec


def _migration_tool():
    # imported as a package module (not spec_from_file_location) so mutmut's
    # test-mapping can associate these tests with tools/migrate_to_knn.py
    import tools.migrate_to_knn

    return tools.migrate_to_knn


# --- ES / OpenSearch: backend failures propagate ---------------------------


def test_es_search_error_propagates():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.side_effect = ConnectionError("cluster unreachable")
    ses = SignatureES(es)
    with pytest.raises(ConnectionError):
        ses.search_single_record(_rec())


def test_es_insert_error_propagates():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.index.side_effect = RuntimeError("index write rejected")
    ses = SignatureES(es)
    with pytest.raises(RuntimeError, match="rejected"):
        ses.insert_single_record(_rec())


def test_es_delete_error_propagates_midway():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {"hits": [{"_id": "keep", "_source": {"path": "p"}}, {"_id": "del", "_source": {"path": "p"}}]}}
    es.delete.side_effect = RuntimeError("delete failed")
    with pytest.raises(RuntimeError, match="delete failed"):
        SignatureES(es).delete_duplicates("p")
    # 'keep' is preserved (skipped), only the duplicate delete was attempted
    es.delete.assert_called_once()
    assert es.delete.call_args.kwargs["id"] == "del"


def test_es_malformed_response_missing_hits():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {}}  # no "hits" list
    with pytest.raises(KeyError):
        SignatureES(es).search_single_record(_rec())


def test_es_hit_missing_signature():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {"hits": [{"_id": "x", "_source": {"path": "p"}}]}}
    with pytest.raises(KeyError):
        SignatureES(es).search_single_record(_rec())


def test_es_record_missing_signature():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = _es_hits()
    rec = _rec()
    del rec["signature"]
    with pytest.raises(KeyError):
        SignatureES(es).search_single_record(rec)


def test_es_delete_duplicates_no_hits():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {"hits": []}}
    SignatureES(es).delete_duplicates("nothing-here")
    es.delete.assert_not_called()


def test_parse_duration_rejects_garbage():
    from image_match.opensearch_driver import _parse_duration

    for bad in ("abc", "", "5h"):
        with pytest.raises(ValueError):
            _parse_duration(bad)


# --- k-NN driver ----------------------------------------------------------


def test_knn_search_error_propagates():
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    client = MagicMock()
    client.search.side_effect = ConnectionError("down")
    with pytest.raises(ConnectionError):
        SignatureOpenSearchKNN(client).search_single_record(_rec())


def test_knn_hit_missing_signature():
    from image_match.opensearch_knn_driver import format_knn_hits

    hits = [{"_id": "x", "_score": 1.0, "_source": {"path": "p"}}]
    with pytest.raises(KeyError):
        format_knn_hits(hits, np.zeros(64), 0.45)


def test_build_knn_query_empty_signature():
    """No client-side validation — the server will reject; document the shape."""
    from image_match.opensearch_knn_driver import build_knn_query

    body = build_knn_query([], k=5)
    assert body["query"]["knn"]["signature"] == {"vector": [], "k": 5}


# --- async drivers ---------------------------------------------------------


async def test_async_es_search_error_propagates():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    es.search = AsyncMock(side_effect=ConnectionError("down"))
    ses = AsyncSignatureES(es)
    with pytest.raises(ConnectionError):
        await ses.search_single_record(_rec())


async def test_async_es_insert_error_propagates():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    es.index = AsyncMock(side_effect=RuntimeError("write failed"))
    with pytest.raises(RuntimeError, match="write failed"):
        await AsyncSignatureES(es).insert_single_record(_rec())


async def test_async_knn_gather_propagates_first_error():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.opensearch_knn_async_driver import AsyncSignatureOpenSearchKNN

    client = _async_es()
    client.search = AsyncMock(side_effect=RuntimeError("boom"))
    ses = AsyncSignatureOpenSearchKNN(client)
    with pytest.raises(RuntimeError, match="boom"):
        await ses.search_image("test.jpg", all_orientations=True)


# --- MongoDB: worker-death contract and timeout ----------------------------


def test_mongo_worker_find_raises_still_enqueues_stop():
    """get_next_match's finally clause must signal completion even on failure —
    otherwise the consumer hangs forever waiting on results_q. The exception
    itself propagates (in production it kills the worker thread)."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    coll.find.side_effect = RuntimeError("cursor died")
    with pytest.raises(RuntimeError, match="cursor died"):
        get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64))
    assert rq.get(timeout=5) == "STOP"
    assert rq.empty()


def test_mongo_worker_doc_missing_signature():
    """A malformed doc kills the worker (KeyError propagates), but the STOP
    sentinel still goes out — the consumer doesn't hang and simply gets
    no match from that word."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    coll.find.return_value.limit.return_value = iter([{"_id": "m1", "path": "p.jpg"}])  # no signature
    with pytest.raises(KeyError):
        get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64))
    assert rq.get(timeout=5) == "STOP"
    assert rq.empty()


def test_mongo_search_process_timeout_returns_partial():
    """A hung worker must not hang the search: after process_timeout the
    consumer stops waiting and returns whatever was collected."""
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 0}])

    def hang(*args, **kwargs):  # worker that never signals
        import time

        time.sleep(60)

    ses = SignatureMongo(coll)
    ses.index_names = ["simple_word_0"]

    import image_match.mongodb_driver as md

    orig = md.get_next_match
    md.get_next_match = hang
    try:
        r = ses.search_single_record(_rec(), process_timeout=0.05, n_parallel_words=1)
    finally:
        md.get_next_match = orig
    assert r == []


def test_mongo_search_empty_index_names():
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection()
    coll.find_one.return_value = None  # no docs -> no word fields discovered
    ses = SignatureMongo(coll)
    assert ses.index_names == []
    assert ses.search_single_record(_rec()) == []


def test_mongo_insert_error_propagates():
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 1}])
    coll.insert_one.side_effect = RuntimeError("insert failed")
    with pytest.raises(RuntimeError, match="insert failed"):
        SignatureMongo(coll).insert_single_record(_rec())


# --- shared helpers: malformed inputs -------------------------------------


def test_dedupe_results_missing_keys():
    from image_match.signature_database_base import dedupe_results

    with pytest.raises(KeyError):
        dedupe_results([{"dist": 0.1}])  # missing "id"
    with pytest.raises(KeyError):
        dedupe_results([{"id": "a"}])  # missing "dist" for the sort


def test_normalized_distance_shape_mismatch():
    from image_match.signature_database_base import normalized_distance

    with pytest.raises(ValueError):
        normalized_distance(np.zeros((1, 4)), np.zeros(7))


def test_search_image_propagates_bad_image():
    from image_match.goldberg import CorruptImageError
    from image_match.signature_database_base import SignatureDatabaseBase

    class Stub(SignatureDatabaseBase):
        def search_single_record(self, rec, pre_filter=None, **kw):
            return []

        def insert_single_record(self, rec, **kw):
            pass

    with pytest.raises(CorruptImageError):
        Stub().search_image(b"not an image", bytestream=True)


def test_search_image_worker_exception_propagates():
    from image_match.signature_database_base import SignatureDatabaseBase

    class Stub(SignatureDatabaseBase):
        def search_single_record(self, rec, pre_filter=None, **kw):
            raise RuntimeError("worker exploded")

        def insert_single_record(self, rec, **kw):
            pass

    with pytest.raises(RuntimeError, match="worker exploded"):
        Stub().search_image("test.jpg", all_orientations=True, n_threads=4)


def test_build_word_query_no_words():
    """A record with no simple_word_* fields yields an empty should list;
    ES interprets that as match-nothing — document the shape."""
    from image_match.elasticsearch_driver import build_word_query

    body = build_word_query({"path": "p", "signature": []})
    should = body["query"]["bool"]["should"]
    assert should == []


# --- migration tool --------------------------------------------------------


def _fake_clients():
    source = MagicMock()
    source.__class__ = type("FakeES", (), {"__module__": "elasticsearch.client"})
    target = MagicMock()
    target.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    target.indices.exists.return_value = False
    target.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    return source, target


def test_migrate_scan_error_propagates(monkeypatch):
    mod = _migration_tool()
    source, target = _fake_clients()

    def exploding_scan(*a, **k):
        yield {"_id": "1", "_source": {"signature": [0.0] * 4, "path": "a"}}
        raise RuntimeError("scroll expired")

    monkeypatch.setattr(mod, "_helpers_for", lambda c: (exploding_scan, lambda c, a: (len(a), [])))
    with pytest.raises(RuntimeError, match="scroll expired"):
        mod.migrate_index(source, target, "src", "dst", dimension=4)


def test_migrate_bulk_errors_not_counted_as_indexed(monkeypatch):
    mod = _migration_tool()
    source, target = _fake_clients()

    hits = [{"_id": str(i), "_source": {"signature": [0.0] * 4, "path": f"{i}.jpg"}} for i in range(3)]

    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter(hits), lambda c, a: (0, [{"error": "rejected"}])))
    stats = mod.migrate_index(source, target, "src", "dst", dimension=4)
    assert stats["scanned"] == 3
    assert stats["indexed"] == 0


def test_verify_index_missing_signature():
    mod = _migration_tool()
    client = MagicMock()
    client.count.return_value = {"count": 1}
    client.search.return_value = {"hits": {"hits": [{"_id": "x", "_source": {"path": "p"}}]}}
    assert mod.verify_index(client, "idx", 648) is False


# --- migration tool: unit coverage for mutation testing ---------------------


def test_detect_server_parsing():
    """Pin the info() parsing: distribution flag, version split, major digit."""
    mod = _migration_tool()

    client = MagicMock()
    client.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    assert mod.detect_server(client) == ("opensearch", 2)
    client.info.assert_called_once_with()

    client.info.return_value = {"version": {"number": "7.17.13"}}
    assert mod.detect_server(client) == ("elasticsearch", 7)

    client.info.return_value = {"version": {"distribution": "opensearch", "number": "3.8.1"}}
    assert mod.detect_server(client) == ("opensearch", 3)

    # no version key at all -> elasticsearch, unknown major
    client.info.return_value = {}
    assert mod.detect_server(client) == ("elasticsearch", None)

    # non-digit version -> unknown major
    client.info.return_value = {"version": {"distribution": "opensearch", "number": "unknown"}}
    assert mod.detect_server(client) == ("opensearch", None)


def test_detect_server_type_wrapper():
    mod = _migration_tool()
    client = MagicMock()
    client.info.return_value = {"version": {"number": "8.0.0"}}
    assert mod.detect_server_type(client) == "elasticsearch"


def test_helpers_for_dispatches_on_client_module():
    """opensearchpy clients get opensearchpy.helpers, others elasticsearch's."""
    mod = _migration_tool()

    # real instances: type() must report the fake client class
    os_client = type("FakeOS", (), {"__module__": "opensearchpy.client"})()
    es_client = type("FakeES", (), {"__module__": "elasticsearch.client"})()

    assert mod._helpers_for(os_client)[0].__module__.startswith("opensearchpy")
    assert mod._helpers_for(es_client)[0].__module__.startswith("elasticsearch")


def test_verify_index_call_args():
    """Pin count/search calls — index name, sample size, signature projection."""
    mod = _migration_tool()
    client = MagicMock()
    client.count.return_value = {"count": 2}
    client.search.return_value = {"hits": {"hits": [{"_id": "x", "_source": {"signature": [0.0] * 4}}, {"_id": "y", "_source": {"signature": [0.0] * 4}}]}}
    assert mod.verify_index(client, "idx", 4) is True
    client.count.assert_called_once_with(index="idx")
    client.search.assert_called_once_with(index="idx", body={"size": 100, "query": {"match_all": {}}}, _source=["signature"])


def test_migrate_index_scan_and_bulk_call_args(monkeypatch):
    """The scan helper must be called with the source client/index and the
    match_all query; bulk with the target client."""
    mod = _migration_tool()
    source, target = _fake_clients()
    hits = [{"_id": "1", "_source": {"signature": [0.0] * 4, "path": "a"}}]

    seen = {}
    helpers_calls = []

    def fake_scan(client, **kw):
        seen["scan_client"] = client
        seen["scan_kwargs"] = kw
        return iter(hits)

    def fake_bulk(client, actions):
        seen["bulk_client"] = client
        seen["actions"] = list(actions)
        return len(actions), []

    def fake_helpers(c):
        helpers_calls.append(c)
        return fake_scan, fake_bulk

    monkeypatch.setattr(mod, "_helpers_for", fake_helpers)
    stats = mod.migrate_index(source, target, "src", "dst", dimension=4)
    assert stats["indexed"] == 1
    assert helpers_calls == [source, target]  # source for scan, target for bulk
    assert seen["scan_client"] is source
    assert seen["bulk_client"] is target
    assert seen["scan_kwargs"] == {
        "index": "src",
        "query": {"query": {"match_all": {}}},
        "_source": True,
        "preserve_order": False,
    }
    # target existence probe and bulk action dict shape are pinned exactly
    target.indices.exists.assert_called_once_with(index="dst")
    assert seen["actions"] == [{"_index": "dst", "_id": "1", "_source": {"signature": [0.0] * 4, "path": "a"}}]


def test_migrate_index_skips_wrong_signature_shapes(monkeypatch):
    """Non-list signatures and wrong dimensions are skipped, not indexed."""
    mod = _migration_tool()
    source, target = _fake_clients()
    hits = [
        {"_id": "ok", "_source": {"signature": [0.0] * 4}},
        {"_id": "short", "_source": {"signature": [0.0] * 3}},
        {"_id": "str", "_source": {"signature": "notalist"}},
        {"_id": "missing", "_source": {}},
    ]
    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter(hits), lambda c, a: (len(a), [])))
    stats = mod.migrate_index(source, target, "src", "dst", dimension=4)
    assert stats == {"scanned": 4, "indexed": 1, "skipped": 3}


def test_migrate_index_nmslib_guard_matrix(monkeypatch):
    """nmslib rejected only on opensearch >= 3 targets when creating the index."""
    mod = _migration_tool()

    def target_with(version):
        t = MagicMock()
        t.indices.exists.return_value = False
        t.info.return_value = {"version": version}
        return t

    source, _ = _fake_clients()
    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter([]), lambda c, a: (0, [])))

    # OS3 + nmslib -> ValueError
    with pytest.raises(ValueError, match=r"\Aengine 'nmslib' is not supported on OpenSearch"):
        mod.migrate_index(source, target_with({"distribution": "opensearch", "number": "3.0.0"}), "s", "d", engine="nmslib")
    # OS2 + nmslib -> allowed (target_type/OS version guards)
    mod.migrate_index(source, target_with({"distribution": "opensearch", "number": "2.19.0"}), "s", "d", engine="nmslib")
    # OS3 + lucene -> allowed (kills 'and' -> 'or' mutant)
    mod.migrate_index(source, target_with({"distribution": "opensearch", "number": "3.0.0"}), "s", "d", engine="lucene")
    # ES + nmslib -> allowed (kills == -> != mutant)
    mod.migrate_index(source, target_with({"number": "7.17.13"}), "s", "d", engine="nmslib")
    # unknown major + nmslib -> allowed (kills 'is not None' drop)
    mod.migrate_index(source, target_with({"distribution": "opensearch", "number": "unknown"}), "s", "d", engine="nmslib")


def test_migrate_index_existing_target_skips_nmslib_check(monkeypatch):
    """Appending into an existing index must not trip the engine guard."""
    mod = _migration_tool()
    source, target = _fake_clients()
    target.indices.exists.return_value = True  # index already exists
    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter([]), lambda c, a: (0, [])))
    # OS3 target + nmslib but index exists -> no error, no create
    stats = mod.migrate_index(source, target, "src", "dst", engine="nmslib")
    assert stats == {"scanned": 0, "indexed": 0, "skipped": 0}
    target.indices.create.assert_not_called()


def test_verify_index_default_sample_size():
    mod = _migration_tool()
    client = MagicMock()
    client.count.return_value = {"count": 0}
    client.search.return_value = {"hits": {"hits": []}}
    assert mod.verify_index(client, "idx", 648) is True
    assert client.search.call_args.kwargs["body"]["size"] == 100


def test_migrate_index_defaults(monkeypatch):
    """Default dimension/engine/space_type/data_type/batch_size flow into the
    created index body and the bulk batching."""
    from image_match.opensearch_knn_driver import knn_index_body

    mod = _migration_tool()
    source, target = _fake_clients()
    hits = [{"_id": str(i), "_source": {"signature": [0.0] * 648, "path": f"{i}.jpg"}} for i in range(501)]
    bulk_sizes = []

    def fake_bulk(client, actions):
        bulk_sizes.append(len(actions))
        return len(actions), []

    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter(hits), fake_bulk))
    stats = mod.migrate_index(source, target, "src", "dst")
    # dimension default 648 -> all docs indexed (a 649 mutant would skip all)
    assert stats == {"scanned": 501, "indexed": 501, "skipped": 0}
    # batch_size default 500 -> flush at 500 plus the remainder
    assert bulk_sizes == [500, 1]
    # engine/space_type/data_type defaults flow into the mapping body
    target.indices.create.assert_called_once_with(index="dst", body=knn_index_body(648, "lucene", "l2", "float"))

    # a non-default data_type must reach the index body (kills the kwarg drop)
    target2 = MagicMock()
    target2.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    target2.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    target2.indices.exists.return_value = False
    mod.migrate_index(source, target2, "src", "dst", data_type="byte")
    target2.indices.create.assert_called_once_with(index="dst", body=knn_index_body(648, "lucene", "l2", "byte"))


def test_main_verify_failure_no_delete_source(monkeypatch):
    """Verification failure -> exit 1 and the source index is left alone."""
    mod = _migration_tool()

    source = MagicMock()
    source.__class__ = type("FakeES", (), {"__module__": "elasticsearch.client"})
    source.info.return_value = {"version": {"number": "7.17.13"}}
    target = MagicMock()
    target.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    target.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    target.indices.exists.return_value = False
    target.count.return_value = {"count": 0}
    target.search.return_value = {"hits": {"hits": [{"_id": "x", "_source": {}}]}}

    monkeypatch.setattr(mod, "make_client", lambda url, st: source if st == "elasticsearch" else target)
    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter([]), lambda c, a: (0, [])))

    rc = mod.main([
        "--source-url",
        "http://src",
        "--source-index",
        "s",
        "--source-type",
        "es",
        "--target-url",
        "http://dst",
        "--target-index",
        "d",
        "--delete-source",
    ])
    assert rc == 1
    source.indices.delete.assert_not_called()


def test_main_argparse_errors():
    mod = _migration_tool()
    with pytest.raises(SystemExit) as e:
        mod.main([])  # missing required args
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        mod.main([
            "--source-url",
            "x",
            "--source-index",
            "s",
            "--target-url",
            "y",
            "--target-index",
            "d",
            "--engine",
            "bogus",
        ])
    assert e.value.code == 2


_BASE_ARGV = [
    "--source-url",
    "http://src",
    "--source-index",
    "s",
    "--source-type",
    "es",
    "--target-url",
    "http://dst",
    "--target-index",
    "d",
]


def _main_clients(mod, monkeypatch):
    """Wire fake source/target clients and spies into the migration tool."""
    source = MagicMock()
    source.__class__ = type("FakeES", (), {"__module__": "elasticsearch.client"})
    source.info.return_value = {"version": {"number": "7.17.13"}}
    target = MagicMock()
    target.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    target.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    target.indices.exists.return_value = False

    monkeypatch.setattr(mod, "make_client", lambda url, st: source if st == "elasticsearch" else target)
    monkeypatch.setattr(mod, "_helpers_for", lambda c: (lambda *a, **k: iter([]), lambda c, a: (0, [])))
    migrate_calls = []
    verify_calls = []
    monkeypatch.setattr(mod, "migrate_index", lambda *a, **kw: migrate_calls.append((a, kw)) or {"scanned": 0, "indexed": 0, "skipped": 0})
    monkeypatch.setattr(mod, "verify_index", lambda *a, **kw: verify_calls.append((a, kw)) or True)
    return source, target, migrate_calls, verify_calls


def test_main_success_path(monkeypatch):
    """rc 0; parsed defaults flow into migrate_index; target is refreshed."""
    mod = _migration_tool()
    source, target, migrate_calls, verify_calls = _main_clients(mod, monkeypatch)

    rc = mod.main(_BASE_ARGV)
    assert rc == 0
    assert len(migrate_calls) == 1
    args, kwargs = migrate_calls[0]
    assert args[0] is source and args[1] is target
    assert args[2:] == ("s", "d")
    assert kwargs == {"dimension": 648, "engine": "lucene", "space_type": "l2", "data_type": "float", "batch_size": 500}
    # verify_index must get (target_client, target_index, dimension)
    assert verify_calls == [((target, "d", 648), {})]
    target.indices.refresh.assert_called_once_with(index="d")
    source.indices.delete.assert_not_called()


def test_main_delete_source(monkeypatch):
    mod = _migration_tool()
    source, _target, _, _ = _main_clients(mod, monkeypatch)
    rc = mod.main([*_BASE_ARGV, "--delete-source"])
    assert rc == 0
    source.indices.delete.assert_called_once_with(index="s")


@pytest.mark.parametrize(
    "dropped",
    ["--source-url", "--source-index", "--target-url", "--target-index"],
)
def test_main_required_args(dropped):
    """Each required flag must be enforced (kills required=False mutants)."""
    mod = _migration_tool()
    argv = [a for i, a in enumerate(_BASE_ARGV) if not (_BASE_ARGV[i - 1] == dropped or a == dropped)]
    with pytest.raises(SystemExit) as e:
        mod.main(argv)
    assert e.value.code == 2


@pytest.mark.parametrize(
    "flag,bad",
    [
        ("--source-type", "bogus"),
        ("--data-type", "bogus"),
        ("--dimension", "notanint"),
        ("--batch-size", "notanint"),
    ],
)
def test_main_arg_validation(flag, bad):
    """choices= and type=int are enforced (kills choices/type mutants)."""
    mod = _migration_tool()
    with pytest.raises(SystemExit) as e:
        mod.main([*_BASE_ARGV, flag, bad])
    assert e.value.code == 2


def test_main_auto_detect_source_type(monkeypatch):
    """Default --source-type auto probes the cluster via an opensearch client."""
    mod = _migration_tool()
    _source, _target, _, _ = _main_clients(mod, monkeypatch)

    # auto-detection probes with an opensearch client first
    probe = MagicMock()
    probe.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    probe.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    made = []
    monkeypatch.setattr(mod, "make_client", lambda url, st: made.append((url, st)) or probe)

    argv = [a for i, a in enumerate(_BASE_ARGV) if _BASE_ARGV[i - 1] != "--source-type" and a != "--source-type"]
    rc = mod.main(argv)
    assert rc == 0
    # probe (opensearch) -> source (opensearch, since probe detects os) -> target
    assert made == [("http://src", "opensearch"), ("http://src", "opensearch"), ("http://dst", "opensearch")]
    probe.close.assert_called_once()


def test_main_detect_es_source_via_probe(monkeypatch):
    """A probe that reports elasticsearch -> source client built as 'elasticsearch'."""
    mod = _migration_tool()
    _, _, _, _ = _main_clients(mod, monkeypatch)

    probe = MagicMock()
    probe.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    probe.info.return_value = {"version": {"number": "7.17.13"}}  # no distribution -> es
    made = []
    monkeypatch.setattr(mod, "make_client", lambda url, st: made.append((url, st)) or probe)

    argv = [a for i, a in enumerate(_BASE_ARGV) if _BASE_ARGV[i - 1] != "--source-type" and a != "--source-type"]
    rc = mod.main(argv)
    assert rc == 0
    assert made[1] == ("http://src", "elasticsearch")


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--engine", "lucene"),
        ("--engine", "faiss"),
        ("--engine", "nmslib"),
        ("--data-type", "float"),
        ("--data-type", "byte"),
        ("--source-type", "es"),
        ("--source-type", "os"),
    ],
)
def test_main_valid_choices_accepted(monkeypatch, flag, value):
    """Every declared choice must parse — a mutated choices list rejects it."""
    mod = _migration_tool()
    _main_clients(mod, monkeypatch)
    argv = [a for i, a in enumerate(_BASE_ARGV) if not (_BASE_ARGV[i - 1] == flag or a == flag)]
    rc = mod.main([*argv, flag, value])
    assert rc == 0


def test_main_explicit_source_type_auto(monkeypatch):
    """'auto' must stay a valid choice even though it's also the default."""
    mod = _migration_tool()
    _main_clients(mod, monkeypatch)
    probe = MagicMock()
    probe.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    probe.info.return_value = {"version": {"number": "7.17.13"}}
    monkeypatch.setattr(mod, "make_client", lambda url, st: probe)

    argv = [a for i, a in enumerate(_BASE_ARGV) if not (_BASE_ARGV[i - 1] == "--source-type" or a == "--source-type")]
    rc = mod.main([*argv, "--source-type", "auto"])
    assert rc == 0
