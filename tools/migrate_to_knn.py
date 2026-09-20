"""Migrate an existing image-match index to a hybrid OpenSearch k-NN index.

Copies documents from an Elasticsearch or OpenSearch index (word-overlap
layout) into a new OpenSearch index whose 'signature' field is mapped as
knn_vector. All other fields keep dynamic mapping, so the classic word
drivers continue to work on the migrated index.

Documents are copied with helpers.scan + helpers.bulk, preserving _id, so
re-running is idempotent. The source index is never modified unless
--delete-source is given after a successful verification.

Usage:
    uv run python tools/migrate_to_knn.py \
        --source-url http://localhost:9200 --source-index images \
        --target-url http://localhost:9201 --target-index images_knn

    # large indexes: fan out with parallel sliced scrolls (e.g. one per shard)
    uv run python tools/migrate_to_knn.py ... --slices 8

Requires image-match plus the relevant client extras
(image-match[elasticsearch] and/or image-match[opensearch]).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


def detect_server(client: Any) -> tuple[str, int | None]:
    """Return (server_type, major_version) based on cluster info()."""
    info = client.info()
    version = info.get("version", {})
    number = str(version.get("number", ""))
    major = int(number.split(".", 1)[0]) if number[:1].isdigit() else None
    server_type = "opensearch" if version.get("distribution") == "opensearch" else "elasticsearch"
    return server_type, major


def detect_server_type(client: Any) -> str:
    """Return 'opensearch' or 'elasticsearch' based on cluster info()."""
    return detect_server(client)[0]


def make_client(url: str, server_type: str, timeout: int = 30) -> Any:
    """Construct a client for the given server type.

    Retries on timeout are enabled — long scroll/bulk requests on a loaded
    cluster occasionally exceed the default 10s read timeout, and a single
    timeout must not kill a multi-hour migration.
    """
    if server_type == "opensearch":
        try:
            from opensearchpy import OpenSearch
        except ImportError as e:
            raise SystemExit("opensearch-py is required: pip install image-match[opensearch]") from e
        return OpenSearch(url, timeout=timeout, retry_on_timeout=True, max_retries=3)
    try:
        from elasticsearch import Elasticsearch
    except ImportError as e:
        raise SystemExit("elasticsearch is required: pip install image-match[elasticsearch]") from e
    try:
        return Elasticsearch(url, request_timeout=timeout, retry_on_timeout=True, max_retries=3)
    except TypeError:  # elasticsearch-py 7.x uses 'timeout'
        return Elasticsearch(url, timeout=timeout, retry_on_timeout=True, max_retries=3)


def _helpers_for(client: Any) -> tuple[Any, Any]:
    """Pick (scan, bulk) helpers matching the client's library."""
    if type(client).__module__.startswith("opensearchpy"):
        from opensearchpy.helpers import bulk, scan
    else:
        from elasticsearch.helpers import bulk, scan
    return scan, bulk


def _bulk_error_types() -> tuple[type[Exception], ...]:
    """Collect the retryable bulk/transport error classes of installed clients.

    Doc-level bulk failures (e.g. a shard briefly unavailable during
    relocation) and transport errors after the client's own retries are
    transient during long migrations — the batch is re-sent as-is, which is
    safe because every action preserves the source _id (idempotent upsert).
    """
    error_types: list[type[Exception]] = []
    for mod_name, attr in (
        ("opensearchpy.helpers", "BulkIndexError"),
        ("opensearchpy", "TransportError"),
        ("elasticsearch.helpers", "BulkIndexError"),
        ("elasticsearch", "TransportError"),
    ):
        try:
            error_types.append(getattr(importlib.import_module(mod_name), attr))
        except (ImportError, AttributeError):
            continue
    return tuple(error_types)


def _resilient_bulk(bulk: Any, client: Any, actions: list[dict[str, Any]], max_attempts: int = 10) -> tuple[int, Any]:
    """helpers.bulk with exponential-backoff retry on transient failures."""
    error_types = _bulk_error_types()
    for attempt in range(max_attempts):
        try:
            return bulk(client, actions)
        except error_types:
            if attempt == max_attempts - 1:
                raise
            time.sleep(min(5 * 2**attempt, 120))
    raise AssertionError("unreachable")


def migrate_index(
    source_client: Any,
    target_client: OpenSearch,
    source_index: str,
    target_index: str,
    dimension: int = 648,
    engine: str = "lucene",
    space_type: str = "l2",
    data_type: str = "float",
    batch_size: int = 500,
    slices: int = 1,
    skip_existing: bool = False,
) -> dict[str, int]:
    """Copy documents from a word-overlap index into a new k-NN index.

    Args:
        source_client: an elasticsearch.Elasticsearch or opensearchpy.OpenSearch client
        target_client: an opensearchpy.OpenSearch client for the destination
        source_index: index name to read from
        target_index: index name to create and write to
        dimension: required signature vector length (648 for default n_grid=9)
        engine: knn engine for the target mapping (default 'lucene');
            'nmslib' is rejected when the target is OpenSearch 3+
        space_type: vector space for the target mapping (default 'l2')
        data_type: knn_vector data type — 'float' (default) or 'byte'
        batch_size: bulk request batch size (default 500)
        slices: number of parallel sliced scrolls (default 1 — sequential).
            >1 fans the scan out across N threads; matching the source index's
            shard count is a reasonable choice. Both clients are thread-safe.
        skip_existing: resume mode — each batch is filtered through an mget on
            the target so already-migrated _ids are not re-indexed. Cheap reads
            instead of expensive k-NN writes; makes restarts after a crash
            nearly free.

    Returns:
        a stats dict {scanned, indexed, skipped}

    """
    from image_match.opensearch_knn_driver import knn_index_body

    scan, _ = _helpers_for(source_client)
    _, target_bulk = _helpers_for(target_client)

    if target_client.indices.exists(index=target_index):
        print(f"target index '{target_index}' already exists — appending into it")
    else:
        target_type, target_major = detect_server(target_client)
        if target_type == "opensearch" and target_major is not None and target_major >= 3 and engine == "nmslib":
            raise ValueError("engine 'nmslib' is not supported on OpenSearch 3+ (blocked for new indexes since 3.0) — use 'lucene' or 'faiss'")
        target_client.indices.create(index=target_index, body=knn_index_body(dimension, engine, space_type, data_type))

    def _migrate_slice(slice_id: int | None) -> dict[str, int]:
        stats = {"scanned": 0, "indexed": 0, "skipped": 0}
        actions: list[dict[str, Any]] = []

        def flush() -> None:
            if not actions:
                return
            batch = actions
            if skip_existing:
                existing = {
                    d["_id"]
                    for d in target_client.mget(index=target_index, body={"ids": [a["_id"] for a in batch]})["docs"]
                    if d["found"]
                }
                stats["skipped"] += len(existing)
                batch = [a for a in batch if a["_id"] not in existing]
            if batch:
                ok, _ = _resilient_bulk(target_bulk, target_client, batch)
                stats["indexed"] += ok
            actions.clear()

        query: dict[str, Any] = {"query": {"match_all": {}}}
        if slice_id is not None:
            query["slice"] = {"id": slice_id, "max": slices}

        for hit in scan(source_client, index=source_index, query=query, _source=True, preserve_order=False):
            stats["scanned"] += 1
            doc = hit["_source"]
            sig = doc.get("signature")
            if not isinstance(sig, list) or len(sig) != dimension:
                stats["skipped"] += 1
                continue
            actions.append({"_index": target_index, "_id": hit["_id"], "_source": doc})
            if len(actions) >= batch_size:
                flush()
        flush()
        return stats

    if slices <= 1:
        return _migrate_slice(None)

    from concurrent.futures import ThreadPoolExecutor

    stats = {"scanned": 0, "indexed": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=slices) as pool:
        for slice_stats in pool.map(_migrate_slice, range(slices)):
            for key in stats:
                stats[key] += slice_stats[key]
    return stats


def verify_index(client: Any, index: str, dimension: int, sample_size: int = 100) -> bool:
    """Check that every sampled doc in `index` carries a signature of `dimension`."""
    count = client.count(index=index)["count"]
    res = client.search(index=index, body={"size": sample_size, "query": {"match_all": {}}}, _source=["signature"])
    hits = res["hits"]["hits"]
    bad = [h["_id"] for h in hits if len(h.get("_source", {}).get("signature") or []) != dimension]
    print(f"verified {index}: {count} docs, sampled {len(hits)}, {len(bad)} with bad/missing signature")
    return not bad


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-url",
        required=True,
        help="e.g. http://localhost:9200",
    )
    parser.add_argument(
        "--source-index",
        required=True,
    )
    parser.add_argument(
        "--source-type",
        choices=["auto", "es", "os"],
        default="auto",
        help="default: auto-detect via cluster info",
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="e.g. http://localhost:9201 (OpenSearch)",
    )
    parser.add_argument(
        "--target-index",
        required=True,
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=648,
        help="signature length (648 for default n_grid=9)",
    )
    parser.add_argument(
        "--engine",
        default="lucene",
        choices=["lucene", "faiss", "nmslib"],
        help="knn engine (default lucene); nmslib is not available on OpenSearch 3+ targets",
    )
    parser.add_argument(
        "--space-type",
        default="l2",
        help="l2, cosinesimil, innerproduct, hamming (engine-dependent)",
    )
    parser.add_argument(
        "--data-type",
        default="float",
        choices=["float", "byte"],
        help="knn_vector data type; 'byte' stores int8 signatures losslessly at 4x smaller footprint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=1,
        help="parallel sliced scrolls (default 1; try matching the source shard count)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="per-request timeout in seconds for both clients (default 30)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="resume mode: mget-filter each batch so already-migrated _ids are not re-indexed",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="delete the source index after successful verification",
    )
    args = parser.parse_args(argv)

    # build the source client, auto-detecting the server type if requested
    if args.source_type == "auto":
        probe = make_client(args.source_url, "opensearch", timeout=args.timeout)
        args.source_type = "os" if detect_server_type(probe) == "opensearch" else "es"
        probe.close()
        print(f"auto-detected source type: {args.source_type}")

    source = make_client(args.source_url, "opensearch" if args.source_type == "os" else "elasticsearch", timeout=args.timeout)
    target = make_client(args.target_url, "opensearch", timeout=args.timeout)

    src_type, src_major = detect_server(source)
    dst_type, dst_major = detect_server(target)
    print(f"migrating {args.source_index} @ {args.source_url} ({src_type} {src_major}) -> {args.target_index} @ {args.target_url} ({dst_type} {dst_major})")
    stats = migrate_index(
        source,
        target,
        args.source_index,
        args.target_index,
        dimension=args.dimension,
        engine=args.engine,
        space_type=args.space_type,
        data_type=args.data_type,
        batch_size=args.batch_size,
        slices=args.slices,
        skip_existing=args.skip_existing,
    )
    print(f"done: {stats}")

    if not verify_index(target, args.target_index, args.dimension):
        print("verification FAILED — target index left in place for inspection", file=sys.stderr)
        return 1

    target.indices.refresh(index=args.target_index)

    if args.delete_source:
        print(f"deleting source index {args.source_index}")
        source.indices.delete(index=args.source_index)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
