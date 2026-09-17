"""Unit tests for the database drivers using mocked clients — no services needed."""

from unittest.mock import AsyncMock, MagicMock, patch

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
    assert "simple_word_0" in rec
    assert isinstance(rec["simple_word_0"], int)
    assert rec["simple_word_0"] >= 0  # base-5 encoded words are non-negative


def test_make_record_no_metadata():
    from image_match.goldberg import ImageSignature

    rec = make_record("test.jpg", ImageSignature(), k=16, N=63)
    assert "metadata" not in rec


def test_get_words_overlapping():
    words = get_words(np.arange(10, dtype=np.int8), k=4, N=5)
    assert words.shape == (5, 4)


def test_get_words_validation():
    with pytest.raises(ValueError, match=r"\AWord length cannot be longer than array length\Z"):
        get_words(np.arange(4), k=10, N=2)
    with pytest.raises(ValueError, match=r"\ANumber of words cannot be more than array length\Z"):
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


def test_normalized_distance_per_row():
    """norms must be computed per row (axis=1), not flattened."""
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = normalized_distance(a, np.array([1.0, 0.0]))
    assert out.shape == (2,)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(np.sqrt(2) / 2)


def test_normalized_distance_no_fp_warnings():
    """0/0 must stay silenced by np.errstate (invalid='ignore')."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalized_distance(np.zeros((2, 3)), np.zeros(3))


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
    # anchored match= patterns: XX-wrapped or case-mutated messages must fail
    with pytest.raises(TypeError, match=r"\Ak should be an integer\Z"):
        SignatureDatabaseBase(k="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"\AN should be an integer\Z"):
        SignatureDatabaseBase(N=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"\An_grid should be an integer\Z"):
        SignatureDatabaseBase(n_grid=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"\Adistance_cutoff should be a float\Z"):
        SignatureDatabaseBase(distance_cutoff="hi")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\Adistance_cutoff should be > 0 \(got -1\)\Z"):
        SignatureDatabaseBase(distance_cutoff=-1)
    # boundary: zero cutoff is valid (only negative is rejected)
    SignatureDatabaseBase(distance_cutoff=0.0)


def test_base_init_signature_args_forwarded():
    """*signature_args reach ImageSignature positionally — but 'n' is always
    bound as a kwarg, so any positional arg currently raises TypeError.
    Pinning that contract: a *args-drop mutant would silently succeed."""
    with pytest.raises(TypeError):
        SignatureDatabaseBase(16, 63, 9, (5, 95), 0.45, None)
    SignatureDatabaseBase(16, 63, 9, (5, 95), 0.45)  # empty args is fine


def test_base_init_signature_kwargs_forwarded():
    base = SignatureDatabaseBase(diagonal_neighbors=False, n_levels=3)
    assert base.gis.diagonal_neighbors is False
    assert base.gis.n_levels == 3


def test_base_abstract_methods():
    base = SignatureDatabaseBase()
    with pytest.raises(NotImplementedError):
        base.search_single_record({})
    with pytest.raises(NotImplementedError):
        base.insert_single_record({})


def test_base_init_defaults():
    base = SignatureDatabaseBase()
    assert base.k == 16
    assert base.N == 63
    assert base.n_grid == 9
    assert base.distance_cutoff == 0.45
    assert base.crop_percentile == (5, 95)


def test_make_record_bytestream_flag():
    """bytestream must be forwarded to generate_signature (kills mutants that
    drop or None-out the kwarg)."""
    from image_match.goldberg import ImageSignature

    with open("test.jpg", "rb") as f:
        data = f.read()
    gis = ImageSignature()
    rec = make_record("x", gis, k=16, N=63, img=data, bytestream=True)
    assert rec["signature"] == gis.generate_signature("test.jpg").tolist()


def test_make_record_img_overrides_path():
    """When img is given, the signature comes from img, not path."""
    from image_match.goldberg import ImageSignature

    gis = ImageSignature()
    other = gis.preprocess_image("test_diff.jpg")
    rec = make_record("test.jpg", gis, k=16, N=63, img=other)
    assert rec["path"] == "test.jpg"
    assert rec["signature"] == gis.generate_signature(other).tolist()


def test_normalized_distance_default_nan_value():
    # zero vs zero -> 0/0 -> default nan_value is 1.0
    assert normalized_distance(np.zeros((1, 3)), np.zeros(3))[0] == 1.0


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

    def insert_records(self, records, **kwargs):
        self.inserted.extend(records)
        return len(records)


def test_add_image_delegates():
    d = _StubDriver()
    d.add_image("test.jpg")
    assert len(d.inserted) == 1
    assert d.inserted[0]["path"] == "test.jpg"


def test_add_image_forwards_insert_args():
    """*args/**kwargs on add_image must reach insert_single_record."""
    d = _StubDriver()
    spy = MagicMock()
    d.insert_single_record = spy
    d.add_image("test.jpg", None, False, None, "EXTRA", timeout=30)
    call = spy.call_args
    assert call.args[1:] == ("EXTRA",)
    assert call.kwargs == {"timeout": 30}


def test_add_image_bytestream_forwarded():
    """bytestream=True must reach make_record — a dropped/None'd flag makes it
    treat the image bytes as a path."""
    d = _StubDriver()
    with open("test.jpg", "rb") as f:
        data = f.read()
    d.add_image("alias", img=data, bytestream=True, metadata={"k": 1})
    assert d.inserted[0]["path"] == "alias"
    assert d.inserted[0]["metadata"] == {"k": 1}
    assert len(d.inserted[0]["signature"]) == 648


def test_add_images_bulk_delegates():
    """add_images generates one record per path and bulk-inserts them."""
    d = _StubDriver()
    n = d.add_images(["test.jpg", "test2.jpg"])
    assert n == 2
    assert [r["path"] for r in d.inserted] == ["test.jpg", "test2.jpg"]
    assert all(len(r["signature"]) == 648 for r in d.inserted)


def test_add_images_threads_and_kwargs():
    """n_threads parallelizes signature gen; **kwargs reach insert_records."""
    d = _StubDriver()
    d.insert_records = MagicMock(return_value=2)
    n = d.add_images(["test.jpg", "test2.jpg"], n_threads=2, refresh_after=True)
    assert n == 2
    recs = d.insert_records.call_args.args[0]
    assert [r["path"] for r in recs] == ["test.jpg", "test2.jpg"]
    assert d.insert_records.call_args.kwargs == {"refresh_after": True}


def test_add_images_metadata_broadcast_and_list():
    """A dict broadcasts to all records; a list aligns per-image."""
    d = _StubDriver()
    recs = d._make_records(["test.jpg", "test2.jpg"], metadata={"t": 1})
    assert [r["metadata"] for r in recs] == [{"t": 1}, {"t": 1}]

    recs = d._make_records(["test.jpg", "test2.jpg"], metadata=[{"a": 1}, None])
    assert recs[0]["metadata"] == {"a": 1}
    assert "metadata" not in recs[1]

    with pytest.raises(ValueError, match="metadata list"):
        d._make_records(["test.jpg"], metadata=[{"a": 1}, {"b": 2}])


def test_add_images_chunked():
    """chunk_size splits generation+insertion; refresh_after lands on the last chunk only."""
    d = _StubDriver()
    d.insert_records = MagicMock(side_effect=lambda recs, **kw: len(recs))
    n = d.add_images(["test.jpg", "test2.jpg", "test.jpg", "test2.jpg", "test.jpg"], chunk_size=2, refresh_after=True)
    assert n == 5
    calls = d.insert_records.call_args_list
    assert [len(c.args[0]) for c in calls] == [2, 2, 1]
    assert [c.kwargs.get("refresh_after") for c in calls] == [False, False, True]


def test_add_images_chunk_size_larger_than_input():
    """chunk_size >= len(paths) is a single bulk call."""
    d = _StubDriver()
    d.insert_records = MagicMock(return_value=2)
    assert d.add_images(["test.jpg", "test2.jpg"], chunk_size=10) == 2
    assert d.insert_records.call_count == 1

    with pytest.raises(ValueError, match="chunk_size"):
        d.add_images(["test.jpg"], chunk_size=0)


def test_add_images_chunked_metadata_aligned():
    """Per-image metadata stays aligned across chunk boundaries."""
    d = _StubDriver()
    n = d.add_images(["test.jpg", "test2.jpg", "test.jpg"], metadata=[{"i": 0}, {"i": 1}, {"i": 2}], chunk_size=2)
    assert n == 3
    assert [r["metadata"]["i"] for r in d.inserted] == [0, 1, 2]


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


def test_search_image_default_single_orientation():
    """Without all_orientations, exactly one record is searched."""
    d = _StubDriver()
    seen = []
    d.search_single_record = lambda rec, pre_filter=None, **kw: seen.append(rec) or []
    d.search_image("test.jpg")
    assert len(seen) == 1


def test_orientation_records_defaults():
    """_orientation_records defaults: single orientation, path input (bytestream
    defaults to False — a str path would raise TypeError otherwise)."""
    d = _StubDriver()
    records = d._orientation_records("test.jpg")
    assert len(records) == 1


def test_orientation_records_inversions(monkeypatch):
    """all_orientations must include the color-inverted variants."""
    import image_match.signature_database_base as base_mod

    d = _StubDriver()
    seen = []
    monkeypatch.setattr(base_mod, "make_record", lambda path, gis, k, N, img=None, **kw: seen.append(img) or {"signature": img})
    records = d._orientation_records("test.jpg", all_orientations=True)
    assert len(records) == 16
    base = seen[0]  # first transform is the identity composition
    assert any(np.array_equal(img, -base) for img in seen[1:])


def test_orientation_records_keep_original_path(monkeypatch):
    """Orientation records must carry the query path, not the transformed
    ndarray (upstream PR #155)."""
    import image_match.signature_database_base as base_mod

    d = _StubDriver()
    seen = []
    monkeypatch.setattr(base_mod, "make_record", lambda path, gis, k, N, img=None, **kw: seen.append(path) or {"path": path})
    d._orientation_records("test.jpg", all_orientations=True)
    assert seen == ["test.jpg"] * 16


def _pool_spy(monkeypatch):
    """Patch ThreadPoolExecutor, returning a list that records max_workers."""
    import image_match.signature_database_base as base_mod

    pools = []
    real_pool = base_mod.ThreadPoolExecutor

    class PoolSpy(real_pool):
        def __init__(self, max_workers=None, *a, **kw):
            pools.append(max_workers)
            super().__init__(max_workers)

    monkeypatch.setattr(base_mod, "ThreadPoolExecutor", PoolSpy)
    return pools


def test_search_image_serial_dispatch_forwards_args(monkeypatch):
    """A single record must stay sequential even with n_threads > 1."""
    d = _StubDriver()
    rec = {"signature": np.zeros(648), "path": "x"}
    monkeypatch.setattr(d, "_orientation_records", lambda *a, **kw: [rec])
    pools = _pool_spy(monkeypatch)

    calls = []
    monkeypatch.setattr(d, "search_single_record", lambda rec, pre_filter=None, **kw: calls.append((rec, pre_filter, kw)) or [])

    pre_filter = {"term": {"a": 1}}
    d.search_image("x", pre_filter=pre_filter, n_threads=4, word_limit=7)

    assert pools == []
    assert calls == [(rec, pre_filter, {"word_limit": 7})]


def test_orientation_records_forwards_bytestream(monkeypatch):
    """bytestream must reach preprocess_image (kills drop-the-kwarg mutants)."""
    import numpy as np

    d = _StubDriver()
    seen = {}
    monkeypatch.setattr(
        d.gis,
        "preprocess_image",
        lambda path, **kw: seen.update(kw) or np.zeros((10, 10)),
    )
    import image_match.signature_database_base as base_mod

    monkeypatch.setattr(base_mod, "make_record", lambda *a, **kw: {"signature": []})
    d._orientation_records("x", bytestream=True)
    assert seen["bytestream"] is True


def test_search_image_parallel_dispatch_forwards_args(monkeypatch):
    """Multiple records with n_threads > 1 must use a pool sized to n_threads."""
    d = _StubDriver()
    recs = [{"signature": np.zeros(648), "path": f"r{i}"} for i in range(2)]
    monkeypatch.setattr(d, "_orientation_records", lambda *a, **kw: recs)
    pools = _pool_spy(monkeypatch)

    calls = []
    monkeypatch.setattr(
        d,
        "search_single_record",
        lambda rec, pre_filter=None, **kw: calls.append((rec, pre_filter, kw)) or [{"id": rec["path"], "dist": 0.1}],
    )

    pre_filter = {"term": {"a": 1}}
    out = d.search_image("x", pre_filter=pre_filter, n_threads=3, word_limit=7)

    assert pools == [3]
    assert sorted(c[0]["path"] for c in calls) == ["r0", "r1"]
    assert all(c[1] is pre_filter and c[2] == {"word_limit": 7} for c in calls)
    assert {h["id"] for h in out} == {"r0", "r1"}


def test_base_init_forwards_to_signature():
    """n_grid/crop_percentile and extra kwargs must reach the ImageSignature —
    kills mutants that drop them from the gis construction."""
    base = SignatureDatabaseBase(n_grid=7, crop_percentile=(10, 90))
    assert base.gis.n == 7
    assert base.gis.crop_percentiles == (10, 90)
    # extra kwargs land on ImageSignature (kills **signature_kwargs drop)
    base2 = SignatureDatabaseBase(diagonal_neighbors=False)
    assert base2.gis.diagonal_neighbors is False
    assert base2.gis.sig_length == 9 * 9 * 4  # 4 neighbors only


def test_add_image_forwards_img_metadata(monkeypatch):
    """img/metadata must reach make_record (kills kwarg-None-ing mutants)."""
    import image_match.signature_database_base as base_mod

    d = _StubDriver()
    captured = {}
    monkeypatch.setattr(base_mod, "make_record", lambda *a, **kw: captured.update(kw) or {"signature": []})
    d.add_image("p.jpg", img=np.zeros((4, 4)), metadata={"m": 1})
    assert captured["img"] is not None
    assert captured["metadata"] == {"m": 1}


def test_add_image_forwards_kwargs_to_insert():
    """Extra kwargs must reach insert_single_record (kills *args/**kwargs drops)."""

    class KwargsDriver(_StubDriver):
        def insert_single_record(self, rec, flag=None, **kw):
            self.inserted.append((rec, flag))

    d = KwargsDriver()
    d.add_image("test.jpg", flag="yes")
    assert d.inserted[0][1] == "yes"


def test_search_image_n_threads_boundary(monkeypatch):
    """n_threads > 1 (not >= 1, not > 2) decides serial vs pool dispatch."""
    d = _StubDriver()
    recs = [{"signature": np.zeros(648), "path": f"r{i}"} for i in range(2)]
    monkeypatch.setattr(d, "_orientation_records", lambda *a, **kw: recs)
    monkeypatch.setattr(d, "search_single_record", lambda rec, pre_filter=None, **kw: [])

    pools = _pool_spy(monkeypatch)
    d.search_image("x", n_threads=1)
    assert pools == []  # n_threads=1 stays serial even with multiple records

    pools = _pool_spy(monkeypatch)
    d.search_image("x", n_threads=2)
    assert pools == [2]  # n_threads=2 is already parallel


def test_orientation_records_rotation_order(monkeypatch):
    """The 4 rotations must be 0/90/180/270 — pin each transformed image."""
    import image_match.signature_database_base as base_mod

    d = _StubDriver()
    seen = []
    monkeypatch.setattr(base_mod, "make_record", lambda path, *a, img=None, **kw: seen.append(np.asarray(img)) or {"signature": []})
    d._orientation_records("test.jpg", all_orientations=True)
    # product(inversions, rotations, mirrors): mirrors vary fastest
    base = seen[0]
    assert np.array_equal(seen[1], np.fliplr(base))
    assert np.array_equal(seen[2], np.rot90(base))
    assert np.array_equal(seen[4], np.rot90(base, 2))
    assert np.array_equal(seen[6], np.rot90(base, 3))
    assert np.array_equal(seen[8], -base)  # inversion after all rotations


def test_get_words_exact_content_and_bounds():
    """Pin word positions (linspace 0..len, endpoint=False) and the inclusive
    k==len / N==len boundaries."""
    words = get_words(np.arange(10), k=4, N=5)
    assert words.tolist() == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
        [8, 9, 0, 0],  # tail zero-padded
    ]
    # boundary: k == len and N == len are legal (kills > -> >= mutants)
    assert get_words(np.arange(5), k=5, N=1).tolist() == [[0, 1, 2, 3, 4]]
    assert get_words(np.arange(5), k=1, N=5).shape == (5, 1)


def test_normalized_distance_casts_to_int():
    """Inputs are truncated to int before the norm — a float-cast mutant
    changes the distance for fractional input."""
    # fractional target: int cast truncates to 0 -> dist 1.0, float keeps it -> ~0.05
    assert normalized_distance(np.array([[0.9, 0.0]]), np.array([1.0, 0.0]))[0] == 1.0
    # fractional vec: same reasoning on the other operand
    assert normalized_distance(np.array([[1.0, 0.0]]), np.array([0.9, 0.0]))[0] == 1.0


def test_max_contrast_fractional():
    """Only positive values become 1 — '> 1' would leave 0.5 untouched."""
    arr = np.array([[0.5, -0.5, 0.0]])
    max_contrast(arr)
    assert arr.tolist() == [[1.0, -1.0, 0.0]]


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


def test_es_ctor_defaults_and_search_call():
    """Pin ctor defaults and the exact es.search kwargs — kills default-value
    and None-ing/dropped-kwarg mutants."""
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = _es_hits()
    ses = SignatureES(es)
    assert ses.index == "images"
    assert ses.timeout == "10s"
    assert ses.size == 100
    assert ses.delete_duplicates_limit == 10000
    assert ses.minimum_should_match is None
    assert ses.use_filter_context is False

    ses.search_single_record(_rec())
    call = es.search.call_args
    assert set(call.kwargs) == {"index", "body", "size", "timeout"}
    assert call.kwargs["index"] == "images"
    assert call.kwargs["size"] == 100
    assert call.kwargs["timeout"] == "10s"
    assert call.kwargs["body"]["query"]["bool"]["should"]


def test_es_kwargs_passthrough_to_base():
    """*args/**kwargs must reach SignatureDatabaseBase (kills *args/**kwargs drops)."""
    from image_match.elasticsearch_driver import SignatureES

    ses = SignatureES(MagicMock(), distance_cutoff=0.3)
    assert ses.distance_cutoff == 0.3
    ses = SignatureES(MagicMock(), "i", "10s", 100, 10000, None, False, 20)
    assert ses.k == 20  # positional *args -> base k


def test_es_delete_duplicates_call_args():
    """Pin the path-match query body and the delete kwargs."""
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {"hits": {"hits": [{"_id": "a", "_source": {"path": "p"}}, {"_id": "b", "_source": {"path": "p"}}]}}
    SignatureES(es).delete_duplicates("p")
    call = es.search.call_args
    assert call.kwargs["body"] == {"query": {"match": {"path": "p"}}}
    assert call.kwargs["index"] == "images"
    es.delete.assert_called_once_with(index="images", id="b")


def test_es_delete_image_removes_all_exact_matches():
    """Unlike delete_duplicates, delete_image removes every exact-path hit."""
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "a", "_source": {"path": "p"}},
                {"_id": "b", "_source": {"path": "p"}},
                {"_id": "c", "_source": {"path": "p2"}},  # fuzzy hit, not exact
                {"_id": "d", "_source": {}},  # legacy doc without path
            ]
        }
    }
    ses = SignatureES(es)
    assert ses.delete_image("p") == 2
    deleted = sorted(c.kwargs["id"] for c in es.delete.call_args_list)
    assert deleted == ["a", "b"]
    assert es.search.call_args.kwargs["size"] == 10000  # delete_duplicates_limit default

    es.delete.reset_mock()
    assert ses.delete_image("p", limit=3) == 2
    assert es.search.call_args.kwargs["size"] == 3


def test_es_insert_records_bulk():
    """insert_records stamps timestamps and ships {index, _source} actions."""
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    fake_helpers = MagicMock()
    fake_helpers.bulk.return_value = (2, [])
    recs = [_rec("a.jpg"), _rec("b.jpg")]
    with patch("image_match.elasticsearch_driver.helpers_module", return_value=fake_helpers):
        assert SignatureES(es, index="idx").insert_records(recs, refresh_after=True) == 2

    call = fake_helpers.bulk.call_args
    assert call.args[0] is es
    actions = call.args[1]
    assert [a["_source"]["path"] for a in actions] == ["a.jpg", "b.jpg"]
    assert all(a["_index"] == "idx" and "timestamp" in a["_source"] for a in actions)
    assert call.kwargs == {"refresh": True}


def test_helpers_module_picks_by_client():
    """helpers_module selects es-py vs opensearch-py helpers by client module."""
    from image_match.elasticsearch_driver import helpers_module

    es_mod = pytest.importorskip("elasticsearch", reason="elasticsearch extra not installed")
    os_mod = pytest.importorskip("opensearchpy", reason="opensearch extra not installed")

    assert helpers_module(es_mod.Elasticsearch("http://localhost:1")).bulk is es_mod.helpers.bulk
    assert helpers_module(os_mod.OpenSearch("http://localhost:1")).bulk is os_mod.helpers.bulk


def test_format_hits_cutoff_boundary():
    """dist == cutoff must be excluded ('<' not '<=')."""
    from image_match.elasticsearch_driver import format_hits

    hits = [{"_id": "x", "_score": 1.0, "_source": {"path": "p", "signature": [0.0, 1.0]}}]
    sig = np.array([1.0, 0.0])
    dist = float(normalized_distance(np.array([[0.0, 1.0]]), sig)[0])
    assert format_hits(hits, sig, dist) == []  # equal to cutoff -> dropped
    assert len(format_hits(hits, sig, dist + 1e-9)) == 1


def test_format_hits_result_keys_and_url_fallback():
    """Pin every result key and the url-over-path fallback."""
    from image_match.elasticsearch_driver import format_hits

    hits = [
        {"_id": "a", "_score": 0.9, "_source": {"path": "p1", "signature": [1.0, 0.0], "metadata": {"m": 1}}},
        {"_id": "b", "_score": 0.8, "_source": {"url": "u2", "path": "p2", "signature": [1.0, 0.0]}},
    ]
    out = format_hits(hits, np.array([1.0, 0.0]), 0.9)
    assert len(out) == 2
    assert out[0] == {"id": "a", "score": 0.9, "metadata": {"m": 1}, "path": "p1", "dist": 0.0}
    # legacy 'url' field wins over 'path'
    assert out[1]["path"] == "u2"
    assert out[1]["metadata"] is None


def test_build_word_query_body_shape():
    from image_match.elasticsearch_driver import build_word_query

    body = build_word_query(dict(_rec()))
    assert body["_source"] == {"excludes": ["simple_word_*"]}
    should = body["query"]["bool"]["should"]
    assert should == [{"term": {f"simple_word_{i}": i * 100}} for i in range(5)]
    # no minimum_should_match unless requested
    assert "minimum_should_match" not in body["query"]["bool"]


def test_build_word_query_filter_context_default_msm():
    """minimum_should_match or 1 -> 1 when unset in filter-context mode."""
    from image_match.elasticsearch_driver import build_word_query

    body = build_word_query(dict(_rec()), use_filter_context=True)
    word_bool = body["query"]["bool"]["filter"][0]["bool"]
    assert word_bool["minimum_should_match"] == 1


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


def test_opensearch_ctor_passthrough():
    """All ctor args must reach SignatureES (kills forwarded-kwarg mutants)."""
    from image_match.opensearch_driver import SignatureOpenSearch

    ses = SignatureOpenSearch(MagicMock(), index="idx", timeout="5s", size=9, delete_duplicates_limit=77, minimum_should_match=2, use_filter_context=True)
    assert ses.index == "idx"
    assert ses.timeout == "5s"
    assert ses.size == 9
    assert ses.delete_duplicates_limit == 77
    assert ses.minimum_should_match == 2
    assert ses.use_filter_context is True
    # defaults
    d = SignatureOpenSearch(MagicMock())
    assert (d.index, d.timeout, d.size, d.delete_duplicates_limit) == ("images", "10s", 100, 10000)
    assert d.minimum_should_match is None
    assert d.use_filter_context is False


def test_opensearch_args_kwargs_passthrough():
    """**kwargs must reach SignatureDatabaseBase; positional *args collide with
    the kwarg-bound SignatureES params — pinning the TypeError kills *args drops."""
    from image_match.opensearch_driver import SignatureOpenSearch

    d = SignatureOpenSearch(MagicMock(), distance_cutoff=0.2)
    assert d.distance_cutoff == 0.2  # **kwargs -> base distance_cutoff
    with pytest.raises(TypeError):
        SignatureOpenSearch(MagicMock(), "idx", "10s", 100, 10000, None, False, 20)


def test_parse_duration_values():
    from image_match.opensearch_driver import _parse_duration

    assert _parse_duration("10s") == 10.0
    assert _parse_duration("500ms") == 0.5
    assert _parse_duration("2m") == 120.0
    assert _parse_duration("10m") == 600.0  # multi-digit: s[:-1] not s[:1]
    assert _parse_duration("3") == 3.0
    assert _parse_duration(10) == 10.0  # non-str input via str()


# --- MongoDB driver --------------------------------------------------------


def _mongo_collection(docs=None, n_indexes=1):
    coll = MagicMock()
    coll.count_documents.return_value = len(docs or [])
    coll.find_one.return_value = (docs or [None])[0]
    coll.index_information.return_value = {f"idx_{i}": {} for i in range(n_indexes)}
    # get_next_match fetches via find().limit()
    coll.find.return_value.limit.return_value = iter([])
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


def test_mongo_init_call_args():
    """Pin the probing calls — count_documents({}) and find_one({})."""
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 1}])
    SignatureMongo(coll)
    coll.count_documents.assert_called_once_with({})
    coll.find_one.assert_called_once_with({})


def test_mongo_init_empty_skips_find_one():
    """count > 0 (not >= 0) gates the find_one probe."""
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[])
    coll.count_documents.return_value = 0
    SignatureMongo(coll)
    coll.find_one.assert_not_called()


def test_mongo_kwargs_passthrough():
    """*args/**kwargs must reach SignatureDatabaseBase (kills *args/**kwargs drops)."""
    from image_match.mongodb_driver import SignatureMongo

    ses = SignatureMongo(_mongo_collection(), distance_cutoff=0.2)
    assert ses.distance_cutoff == 0.2
    ses = SignatureMongo(_mongo_collection(), 20)  # positional *args -> base k
    assert ses.k == 20


def test_get_next_match_result_dict_keys():
    """Result dicts must carry dist/path/id/metadata."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    doc = {"_id": "m1", "signature": np.ones(64).tolist(), "path": "p.jpg", "metadata": {"t": 1}}
    coll.find.return_value.limit.return_value = iter([doc])
    get_next_match(rq, {"simple_word_0": 5}, coll, np.ones(64))  # dist 0 < cutoff
    hit = rq.get()["m1"]
    assert set(hit) == {"dist", "path", "id", "metadata"}
    assert hit["path"] == "p.jpg"
    assert hit["metadata"] == {"t": 1}
    assert rq.get() == "STOP"


def test_get_next_match_dist_cutoff_boundary():
    """dist == cutoff is excluded ('<' not '<=')."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    # identical signature -> dist 0.0; cutoff 0.0 must still exclude it
    doc = {"_id": "m1", "signature": np.ones(64).tolist(), "path": "p.jpg"}
    coll.find.return_value.limit.return_value = iter([doc])
    get_next_match(rq, {"simple_word_0": 5}, coll, np.ones(64), cutoff=0.0)
    assert rq.get() == "STOP"  # no result enqueued


def test_get_next_match_cursor_boundary_result():
    """Exactly max_in_cursor docs must be processed (not skipped as popular)."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    docs = [{"_id": f"m{i}", "signature": np.ones(64).tolist(), "path": "p"} for i in range(4)]
    rq = Queue()
    coll = MagicMock()
    coll.find.return_value.limit.return_value = iter(docs)
    get_next_match(rq, {"simple_word_0": 5}, coll, np.ones(64), max_in_cursor=4)
    assert rq.get() == {"m0": {"dist": 0.0, "path": "p", "id": "m0", "metadata": None}}
    rq = Queue()
    get_next_match(rq, {"simple_word_0": 5}, coll, np.ones(64), max_in_cursor=3)
    assert rq.get() == "STOP"  # 4 docs > 3 -> word skipped entirely


def test_mongo_index_collection_args():
    """create_index must be called with each discovered word field."""
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 1, "simple_word_5": 2}])
    ses = SignatureMongo(coll)
    ses.index_collection()
    assert sorted(c.args[0] for c in coll.create_index.call_args_list) == ["simple_word_0", "simple_word_5"]


def test_get_next_match_find_call_args():
    """Pin the query projection and the max_in_cursor+1 limit."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    coll.find.return_value.limit.return_value = iter([])
    get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64))
    coll.find.assert_called_once_with({"simple_word_0": 5}, projection=["_id", "signature", "path", "metadata"])
    coll.find.return_value.limit.assert_called_once_with(101)


def test_get_next_match_at_cursor_boundary():
    """Exactly max_in_cursor docs is fine; +1 triggers the popular-word skip."""
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    docs = [{"_id": f"m{i}", "signature": np.ones(64).tolist()} for i in range(4)]
    rq = Queue()
    coll = MagicMock()
    coll.find.return_value.limit.return_value = iter(docs)
    get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64), max_in_cursor=4)
    assert rq.get() == "STOP"  # boundary reached, not skipped

    rq = Queue()
    coll.find.return_value.limit.return_value = iter(docs)
    get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64), max_in_cursor=3)
    assert rq.get() == "STOP"  # still terminates on the skip path


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


def test_mongo_insert_records_and_delete_image():
    """insert_records uses insert_many + stamps timestamps; delete_image uses
    delete_many on the exact path."""
    from image_match.mongodb_driver import SignatureMongo

    coll = _mongo_collection(docs=[{"simple_word_0": 1, "simple_word_9": 2}])
    ses = SignatureMongo(coll)
    ses.index_names = ["simple_word_0", "simple_word_9"]

    inserted = MagicMock()
    inserted.inserted_ids = ["x", "y"]
    coll.insert_many.return_value = inserted
    recs = [_rec("a.jpg"), _rec("b.jpg")]
    assert ses.insert_records(recs) == 2
    sent = coll.insert_many.call_args.args[0]
    assert [r["path"] for r in sent] == ["a.jpg", "b.jpg"]
    assert all("timestamp" in r for r in sent)
    assert coll.create_index.call_count == 2  # lazy index creation on first bulk

    coll.delete_many.return_value = MagicMock(deleted_count=3)
    assert ses.delete_image("p") == 3
    coll.delete_many.assert_called_once_with({"path": "p"})


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
    coll.find.return_value.limit.return_value = iter([doc])
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
    # word matches more than maximum_matches docs -> skipped as non-discriminatory
    coll.find.return_value.limit.return_value = iter([{"_id": i} for i in range(101)])
    ses = SignatureMongo(coll)
    ses.index_names = ["simple_word_0"]
    assert ses.search_single_record(_rec(), maximum_matches=100) == []
    coll.find.assert_called_once()


def test_get_next_match_pre_filter():
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    coll.find.return_value.limit.return_value = iter([])
    get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64), pre_filter={"metadata.t": "x"})
    # query merged filter + word
    assert coll.find.call_args[0][0] == {"simple_word_0": 5, "metadata.t": "x"}
    assert rq.get() == "STOP"  # sentinel always enqueued


def test_get_next_match_defaults():
    from queue import Queue

    from image_match.mongodb_driver import get_next_match

    rq = Queue()
    coll = MagicMock()
    # doc signature all-ones vs query zeros -> dist = 1.0 > default cutoff 0.5,
    # so it must be filtered out (a cutoff=1.5 mutant would keep it)
    coll.find.return_value.limit.return_value = iter([{"_id": "m1", "signature": np.ones(64).tolist()}])
    get_next_match(rq, {"simple_word_0": 5}, coll, np.zeros(64))
    # driver fetches max_in_cursor + 1 docs to detect overflow (default 100 + 1)
    coll.find.return_value.limit.assert_called_once_with(101)
    assert rq.get() == "STOP"
    assert rq.empty()


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


async def test_async_es_ctor_defaults_and_search_call():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    ses = AsyncSignatureES(es)
    assert ses.index == "images"
    assert ses.timeout == "10s"
    assert ses.size == 100
    assert ses.delete_duplicates_limit == 10000
    assert ses.minimum_should_match is None
    assert ses.use_filter_context is False

    await ses.search_single_record(_rec())
    call = es.search.await_args
    assert set(call.kwargs) == {"index", "body", "size", "timeout"}
    assert call.kwargs["index"] == "images"
    assert call.kwargs["size"] == 100
    assert call.kwargs["timeout"] == "10s"
    assert call.kwargs["body"]["query"]["bool"]["should"]  # body must be a real query


async def test_async_es_args_kwargs_passthrough():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    # *args/**kwargs reach SignatureDatabaseBase directly (no intermediate binding)
    ses = AsyncSignatureES(_async_es(), "idx", "10s", 100, 10000, None, False, 20, distance_cutoff=0.2)
    assert ses.k == 20
    assert ses.distance_cutoff == 0.2


async def test_async_es_delete_duplicates_call_args():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    es.search = AsyncMock(return_value={"hits": {"hits": [{"_id": "a", "_source": {"path": "p"}}, {"_id": "b", "_source": {"path": "p"}}]}})
    ses = AsyncSignatureES(es)
    await ses.delete_duplicates("p")
    call = es.search.await_args
    assert call.kwargs["body"] == {"query": {"match": {"path": "p"}}}
    assert call.kwargs["index"] == "images"
    assert call.kwargs["size"] == 10000  # default delete_duplicates_limit
    es.delete.assert_awaited_once_with(index="images", id="b")


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


async def test_async_es_insert_records_and_delete_image():
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    ses = AsyncSignatureES(es, index="idx")

    fake_helpers = MagicMock()
    fake_helpers.async_bulk = AsyncMock(return_value=(2, []))
    with patch("image_match.elasticsearch_async_driver.helpers_module", return_value=fake_helpers):
        assert await ses.insert_records([_rec("a.jpg"), _rec("b.jpg")], refresh_after=True) == 2
    call = fake_helpers.async_bulk.await_args
    actions = call.args[1]
    assert [a["_source"]["path"] for a in actions] == ["a.jpg", "b.jpg"]
    assert all(a["_index"] == "idx" and "timestamp" in a["_source"] for a in actions)
    assert call.kwargs == {"refresh": True}

    # add_images offloads record generation and forwards kwargs
    fake_helpers.async_bulk.reset_mock()
    with patch("image_match.elasticsearch_async_driver.helpers_module", return_value=fake_helpers):
        n = await ses.add_images(["test.jpg", "test2.jpg"], metadata={"t": 1})
    assert n == 2
    actions = fake_helpers.async_bulk.await_args.args[1]
    assert all(a["_source"]["metadata"] == {"t": 1} for a in actions)

    es.search = AsyncMock(
        return_value={
            "hits": {"hits": [{"_id": "a", "_source": {"path": "p"}}, {"_id": "b", "_source": {"path": "p"}}, {"_id": "c", "_source": {"path": "other"}}]}
        }
    )
    assert await ses.delete_image("p") == 2
    assert es.delete.await_count == 2


async def test_async_es_add_images_chunked():
    """chunk_size splits into multiple async_bulk calls; refresh only on the last."""
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.elasticsearch_async_driver import AsyncSignatureES

    es = _async_es()
    ses = AsyncSignatureES(es, index="idx")

    fake_helpers = MagicMock()
    fake_helpers.async_bulk = AsyncMock(side_effect=lambda client, actions, **kw: (len(actions), []))
    with patch("image_match.elasticsearch_async_driver.helpers_module", return_value=fake_helpers):
        n = await ses.add_images(["test.jpg", "test2.jpg", "test.jpg"], chunk_size=2, refresh_after=True)
    assert n == 3
    calls = fake_helpers.async_bulk.await_args_list
    assert len(calls) == 2
    assert [c.kwargs.get("refresh") for c in calls] == [False, True]


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


async def test_async_opensearch_ctor_passthrough():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.opensearch_async_driver import AsyncSignatureOpenSearch

    ses = AsyncSignatureOpenSearch(_async_es(), index="idx", timeout="5s", size=9, delete_duplicates_limit=77, minimum_should_match=2, use_filter_context=True)
    assert ses.index == "idx"
    assert ses.timeout == "5s"
    assert ses.size == 9
    assert ses.delete_duplicates_limit == 77
    assert ses.minimum_should_match == 2
    assert ses.use_filter_context is True
    # defaults
    d = AsyncSignatureOpenSearch(_async_es())
    assert (d.index, d.timeout, d.size, d.delete_duplicates_limit) == ("images", "10s", 100, 10000)
    assert d.minimum_should_match is None
    assert d.use_filter_context is False

    # **kwargs reach the base class; positional *args collide -> TypeError
    d = AsyncSignatureOpenSearch(_async_es(), distance_cutoff=0.2)
    assert d.distance_cutoff == 0.2
    with pytest.raises(TypeError):
        AsyncSignatureOpenSearch(_async_es(), "i", "10s", 100, 10000, None, False, 20)


async def test_async_knn_ctor_and_search_call():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    pytest.importorskip("aiohttp", reason="async extras not installed")
    from image_match.opensearch_knn_async_driver import AsyncSignatureOpenSearchKNN

    client = _async_es()
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    ses = AsyncSignatureOpenSearchKNN(client)
    assert (ses.index, ses.timeout, ses.size, ses.delete_duplicates_limit) == ("images", "10s", 100, 10000)

    await ses.search_single_record(_rec())
    call = client.search.await_args
    assert call.kwargs["index"] == "images"
    assert call.kwargs["params"] == {"request_timeout": 10.0}
    assert call.kwargs["body"]["query"]["knn"]["signature"]["k"] == 100

    ses2 = AsyncSignatureOpenSearchKNN(_async_es(), distance_cutoff=0.2)
    assert ses2.distance_cutoff == 0.2  # kills **kwargs drop
    ses2 = AsyncSignatureOpenSearchKNN(_async_es(), "i", "10s", 100, 10000, 20)
    assert ses2.k == 20  # *args -> base k


# --- OpenSearch k-NN driver -------------------------------------------------


def test_knn_ctor_defaults_and_search_call():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    client = MagicMock()
    client.search.return_value = {"hits": {"hits": []}}
    ses = SignatureOpenSearchKNN(client)
    assert ses.index == "images"
    assert ses.timeout == "10s"
    assert ses.size == 100
    assert ses.delete_duplicates_limit == 10000

    ses.search_single_record(_rec())
    call = client.search.call_args
    assert set(call.kwargs) == {"index", "body", "params"}
    assert call.kwargs["index"] == "images"
    assert call.kwargs["params"] == {"request_timeout": 10.0}
    knn = call.kwargs["body"]["query"]["knn"]["signature"]
    assert knn["k"] == 100  # size forwarded as k
    assert all(type(v) is float for v in knn["vector"])  # dtype=float, not None
    assert call.kwargs["body"]["size"] == 100
    assert call.kwargs["body"]["_source"] == {"excludes": ["simple_word_*"]}


def test_knn_search_pre_filter_and_kwargs_passthrough():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    client = MagicMock()
    client.search.return_value = {"hits": {"hits": []}}
    ses = SignatureOpenSearchKNN(client, distance_cutoff=0.2)  # kills **kwargs drop
    assert ses.distance_cutoff == 0.2
    ses = SignatureOpenSearchKNN(client, "i", "10s", 100, 10000, 20)  # *args -> base k
    assert ses.k == 20
    ses.search_single_record(_rec(), pre_filter={"term": {"metadata.t": "x"}})
    knn = client.search.call_args.kwargs["body"]["query"]["knn"]["signature"]
    assert knn["filter"] == {"term": {"metadata.t": "x"}}

    ses.search_single_record(_rec(), pre_filter=[{"term": {"a": 1}}, {"term": {"b": 2}}])
    knn = client.search.call_args.kwargs["body"]["query"]["knn"]["signature"]
    assert knn["filter"] == {"bool": {"filter": [{"term": {"a": 1}}, {"term": {"b": 2}}]}}


def test_knn_insert_delete_and_path_hits_call_args():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    client = MagicMock()
    ses = SignatureOpenSearchKNN(client)
    rec = _rec()
    ses.insert_single_record(rec, refresh_after=True)
    call = client.index.call_args
    assert call.kwargs["index"] == "images"
    assert call.kwargs["body"] is rec
    assert call.kwargs["params"] == {"refresh": "true"}

    client.search.return_value = {"hits": {"hits": [{"_id": "a", "_source": {"path": "p"}}, {"_id": "b", "_source": {"path": "p"}}]}}
    ses.delete_duplicates("p")
    call = client.search.call_args
    assert call.kwargs["body"] == {"query": {"match": {"path": "p"}}}
    assert call.kwargs["index"] == "images"
    assert call.kwargs["size"] == 10000
    client.delete.assert_called_once_with(index="images", id="b")


def test_knn_index_body_exact():
    """Pin the full mapping body — kills key-name and structure mutants."""
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import knn_index_body

    assert knn_index_body(648, "lucene", "l2", "float") == {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "signature": {
                    "type": "knn_vector",
                    "dimension": 648,
                    "data_type": "float",
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"},
                }
            }
        },
    }
    assert knn_index_body(4, "faiss", "cosinesimil", "byte")["mappings"]["properties"]["signature"]["method"] == {
        "name": "hnsw",
        "engine": "faiss",
        "space_type": "cosinesimil",
    }


def test_format_knn_hits_rescoring():
    pytest.importorskip("opensearchpy", reason="opensearch extra not installed")
    from image_match.opensearch_knn_driver import format_knn_hits

    sig = np.array([1.0, 0.0])
    hits = [
        {"_id": "near", "_score": 0.9, "_source": {"path": "a", "signature": [1.0, 0.0], "metadata": {"m": 1}}},
        {"_id": "far", "_score": 0.1, "_source": {"path": "b", "signature": [0.0, 1.0]}},
        {"_id": "legacy", "_score": 0.8, "_source": {"url": "u", "path": "ignored", "signature": [1.0, 0.0]}},
    ]
    out = format_knn_hits(hits, sig, 0.45)
    assert [h["id"] for h in out] == ["near", "legacy"]
    assert out[0] == {"id": "near", "score": 0.9, "metadata": {"m": 1}, "path": "a", "dist": 0.0}
    # legacy 'url' field wins over 'path'
    assert out[1]["path"] == "u"
    assert out[1]["metadata"] is None
    # dist == cutoff is excluded
    assert format_knn_hits(hits, sig, 0.0) == []
    assert format_knn_hits([], sig, 0.45) == []


# --- Search tuning: minimum_should_match / filter context -----------------


def test_build_word_query_minimum_should_match():
    from image_match.elasticsearch_driver import build_word_query

    body = build_word_query(dict(_rec()), minimum_should_match=3)
    boolq = body["query"]["bool"]
    assert len(boolq["should"]) == 5
    assert boolq["minimum_should_match"] == 3


def test_build_word_query_minimum_should_match_string():
    from image_match.elasticsearch_driver import build_word_query

    body = build_word_query(dict(_rec()), minimum_should_match="2<75%")
    assert body["query"]["bool"]["minimum_should_match"] == "2<75%"


def test_build_word_query_filter_context():
    from image_match.elasticsearch_driver import build_word_query

    body = build_word_query(
        dict(_rec()),
        pre_filter={"term": {"metadata.t": "x"}},
        minimum_should_match=2,
        use_filter_context=True,
    )
    filters = body["query"]["bool"]["filter"]
    assert filters[0] == {"term": {"metadata.t": "x"}}
    word_bool = filters[1]["bool"]
    assert len(word_bool["should"]) == 5
    assert word_bool["minimum_should_match"] == 2
    # no scoring should-clause at the top level
    assert "should" not in body["query"]["bool"]


def test_build_word_query_filter_context_list_pre_filter():
    from image_match.elasticsearch_driver import build_word_query

    clauses = [{"term": {"metadata.t": "x"}}, {"term": {"metadata.u": "y"}}]
    body = build_word_query(dict(_rec()), pre_filter=clauses, use_filter_context=True)
    filters = body["query"]["bool"]["filter"]
    assert filters[:2] == clauses


def test_es_driver_plumbs_query_options():
    from image_match.elasticsearch_driver import SignatureES

    es = MagicMock()
    es.search.return_value = _es_hits()
    ses = SignatureES(es, minimum_should_match=4, use_filter_context=True)
    ses.search_single_record(_rec())
    body = es.search.call_args.kwargs["body"]
    assert "filter" in body["query"]["bool"]


# --- search_image n_threads ------------------------------------------------


def test_search_image_n_threads():
    import threading

    d = _StubDriver()
    seen_threads = set()
    orig = d.search_single_record

    def spy(rec, pre_filter=None, **kw):
        seen_threads.add(threading.current_thread().name)
        return orig(rec, pre_filter=pre_filter, **kw)

    d.search_single_record = spy
    d.search_image("test.jpg", all_orientations=True, n_threads=4)
    assert len(seen_threads) > 1


def test_search_image_n_threads_default_sequential():
    import threading

    d = _StubDriver()
    seen_threads = set()

    def spy(rec, pre_filter=None, **kw):
        seen_threads.add(threading.current_thread().name)
        return []

    d.search_single_record = spy
    d.search_image("test.jpg", all_orientations=True)
    assert len(seen_threads) == 1
