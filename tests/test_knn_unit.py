"""Unit tests for the OpenSearch k-NN drivers and the migration tool — no services needed."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from .test_drivers_unit import _async_es, _es_hits, _rec

# --- OpenSearch k-NN driver ------------------------------------------------


def test_knn_index_body():
    from image_match.opensearch_knn_driver import knn_index_body

    body = knn_index_body(648)
    assert body["settings"]["index"]["knn"] is True
    sig = body["mappings"]["properties"]["signature"]
    assert sig["type"] == "knn_vector"
    assert sig["dimension"] == 648
    assert sig["data_type"] == "float"
    assert sig["method"]["engine"] == "lucene"
    assert sig["method"]["space_type"] == "l2"

    custom = knn_index_body(64, engine="faiss", space_type="cosinesimil", data_type="byte")
    sig = custom["mappings"]["properties"]["signature"]
    assert sig["dimension"] == 64
    assert sig["data_type"] == "byte"
    assert sig["method"]["engine"] == "faiss"


def test_build_knn_query_shape():
    from image_match.opensearch_knn_driver import build_knn_query

    body = build_knn_query([0.1] * 648, k=7)
    knn = body["query"]["knn"]["signature"]
    assert knn["k"] == 7
    assert len(knn["vector"]) == 648
    assert body["size"] == 7
    assert "filter" not in knn


def test_build_knn_query_pre_filter():
    from image_match.opensearch_knn_driver import build_knn_query

    body = build_knn_query([0.0] * 8, k=3, pre_filter={"term": {"metadata.t": "x"}})
    assert body["query"]["knn"]["signature"]["filter"] == {"term": {"metadata.t": "x"}}

    body = build_knn_query([0.0] * 8, k=3, pre_filter=[{"term": {"a": 1}}, {"term": {"b": 2}}])
    f = body["query"]["knn"]["signature"]["filter"]
    assert f == {"bool": {"filter": [{"term": {"a": 1}}, {"term": {"b": 2}}]}}


def test_knn_search_rescored_by_normalized_distance():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    # knn l2 ranking may differ from normalized-distance ranking; rescore wins
    sig = np.arange(64) % 3 - 1
    client = MagicMock()
    client.search.return_value = _es_hits(n=1, signature=sig.tolist())
    ses = SignatureOpenSearchKNN(client, index="idx", size=9)
    rec = _rec()
    rec["signature"] = sig.tolist()
    r = ses.search_single_record(rec)
    assert len(r) == 1
    assert r[0]["dist"] == 0.0
    call = client.search.call_args
    assert call.kwargs["index"] == "idx"
    assert "knn" in call.kwargs["body"]["query"]
    assert call.kwargs["body"]["query"]["knn"]["signature"]["k"] == 9


def test_knn_insert_and_delete_duplicates():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    client = MagicMock()
    ses = SignatureOpenSearchKNN(client)
    rec = _rec()
    ses.insert_single_record(rec, refresh_after=True)
    call = client.index.call_args
    assert call.kwargs["body"] is rec
    assert call.kwargs["params"]["refresh"] == "true"
    assert "timestamp" in rec

    client.search.return_value = {"hits": {"hits": [{"_id": "keep", "_source": {"path": "p"}}, {"_id": "del", "_source": {"path": "p"}}]}}
    ses.delete_duplicates("p")
    client.delete.assert_called_once()
    assert client.delete.call_args.kwargs["id"] == "del"


async def test_async_knn_search_and_insert():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.opensearch_knn_async_driver import AsyncSignatureOpenSearchKNN

    client = _async_es()
    ses = AsyncSignatureOpenSearchKNN(client, index="idx", size=5)
    sig = np.arange(64) % 3 - 1
    client.search = AsyncMock(return_value=_es_hits(n=1, signature=sig.tolist()))
    rec = _rec()
    rec["signature"] = sig.tolist()
    r = await ses.search_single_record(rec)
    assert r[0]["dist"] == 0.0
    call = client.search.await_args
    assert call.kwargs["body"]["query"]["knn"]["signature"]["k"] == 5
    assert call.kwargs["params"]["request_timeout"] == 10.0


async def test_async_knn_search_image_gathers_orientations():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.opensearch_knn_async_driver import AsyncSignatureOpenSearchKNN

    client = _async_es()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    ses = AsyncSignatureOpenSearchKNN(client)
    await ses.search_image("test.jpg", all_orientations=True)
    # 16 orientation records -> 16 awaited searches
    assert client.search.await_count == 16


# --- migration tool --------------------------------------------------------


def _load_migration_tool():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("migrate_to_knn", Path(__file__).parent.parent / "tools" / "migrate_to_knn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migrate_index_copies_and_validates():
    mod = _load_migration_tool()

    good = {"signature": [0.0] * 4, "path": "a.jpg"}
    bad = {"signature": [0.0] * 3, "path": "b.jpg"}

    source = MagicMock()
    source.__class__ = type("FakeES", (), {"__module__": "elasticsearch.client"})
    target = MagicMock()
    target.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    target.indices.exists.return_value = False

    scan_hits = [{"_id": "1", "_source": good}, {"_id": "2", "_source": bad}]

    # patch helpers selection: fake scan/bulk injected via monkeypatched _helpers_for
    captured = {}

    def fake_scan(client, index, query, _source, preserve_order):
        return iter(scan_hits)

    def fake_bulk(client, actions):
        captured["actions"] = list(actions)
        return len(captured["actions"]), []

    orig = mod._helpers_for
    mod._helpers_for = lambda c: (fake_scan, fake_bulk)
    try:
        stats = mod.migrate_index(source, target, "src", "dst", dimension=4, batch_size=2)
    finally:
        mod._helpers_for = orig

    assert stats == {"scanned": 2, "indexed": 1, "skipped": 1}
    assert captured["actions"][0]["_id"] == "1"
    target.indices.create.assert_called_once()
    assert target.indices.create.call_args.kwargs["index"] == "dst"


def test_detect_server_parses_major_version():
    mod = _load_migration_tool()
    client = MagicMock()
    client.info.return_value = {"version": {"distribution": "opensearch", "number": "3.2.1"}}
    assert mod.detect_server(client) == ("opensearch", 3)

    client.info.return_value = {"version": {"number": "7.17.13"}}
    assert mod.detect_server(client) == ("elasticsearch", 7)

    client.info.return_value = {}
    assert mod.detect_server(client) == ("elasticsearch", None)


def test_migrate_index_rejects_nmslib_on_opensearch3():
    mod = _load_migration_tool()

    source = MagicMock()
    source.__class__ = type("FakeES", (), {"__module__": "elasticsearch.client"})
    target = MagicMock()
    target.__class__ = type("FakeOS", (), {"__module__": "opensearchpy.client"})
    target.indices.exists.return_value = False
    target.info.return_value = {"version": {"distribution": "opensearch", "number": "3.0.0"}}

    with pytest.raises(ValueError, match="nmslib"):
        mod.migrate_index(source, target, "src", "dst", engine="nmslib")
    target.indices.create.assert_not_called()

    # nmslib is still allowed when the target is OpenSearch 2
    target.info.return_value = {"version": {"distribution": "opensearch", "number": "2.19.0"}}
    mod._helpers_for = lambda c: (lambda *a, **k: iter([]), lambda c, a: (0, []))
    mod.migrate_index(source, target, "src", "dst", engine="nmslib")
    target.indices.create.assert_called_once()
