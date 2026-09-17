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

Requires image-match plus the relevant client extras
(image-match[elasticsearch] and/or image-match[opensearch]).
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


def detect_server_type(client: Any) -> str:
    """Return 'opensearch' or 'elasticsearch' based on cluster info()."""
    info = client.info()
    if info.get("version", {}).get("distribution") == "opensearch":
        return "opensearch"
    return "elasticsearch"


def make_client(url: str, server_type: str) -> Any:
    """Construct a client for the given server type."""
    if server_type == "opensearch":
        try:
            from opensearchpy import OpenSearch
        except ImportError as e:
            raise SystemExit("opensearch-py is required: pip install image-match[opensearch]") from e
        return OpenSearch(url)
    try:
        from elasticsearch import Elasticsearch
    except ImportError as e:
        raise SystemExit("elasticsearch is required: pip install image-match[elasticsearch]") from e
    return Elasticsearch(url)


def _helpers_for(client: Any) -> tuple[Any, Any]:
    """Pick (scan, bulk) helpers matching the client's library."""
    if type(client).__module__.startswith("opensearchpy"):
        from opensearchpy.helpers import bulk, scan
    else:
        from elasticsearch.helpers import bulk, scan
    return scan, bulk


def migrate_index(
    source_client: Any,
    target_client: OpenSearch,
    source_index: str,
    target_index: str,
    dimension: int = 648,
    engine: str = "lucene",
    space_type: str = "l2",
    batch_size: int = 500,
) -> dict[str, int]:
    """Copy documents from a word-overlap index into a new k-NN index.

    Args:
        source_client: an elasticsearch.Elasticsearch or opensearchpy.OpenSearch client
        target_client: an opensearchpy.OpenSearch client for the destination
        source_index: index name to read from
        target_index: index name to create and write to
        dimension: required signature vector length (648 for default n_grid=9)
        engine: knn engine for the target mapping (default 'lucene')
        space_type: vector space for the target mapping (default 'l2')
        batch_size: bulk request batch size (default 500)

    Returns:
        a stats dict {scanned, indexed, skipped}

    """
    from image_match.opensearch_knn_driver import knn_index_body

    scan, _ = _helpers_for(source_client)
    _, target_bulk = _helpers_for(target_client)

    if target_client.indices.exists(index=target_index):
        print(f"target index '{target_index}' already exists — appending into it")
    else:
        target_client.indices.create(index=target_index, body=knn_index_body(dimension, engine, space_type))

    stats = {"scanned": 0, "indexed": 0, "skipped": 0}
    actions: list[dict[str, Any]] = []

    def flush() -> None:
        if not actions:
            return
        ok, _ = target_bulk(target_client, actions)
        stats["indexed"] += ok
        actions.clear()

    for hit in scan(source_client, index=source_index, query={"query": {"match_all": {}}}, _source=True, preserve_order=False):
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-url", required=True, help="e.g. http://localhost:9200")
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--source-type", choices=["auto", "es", "os"], default="auto", help="default: auto-detect via cluster info")
    parser.add_argument("--target-url", required=True, help="e.g. http://localhost:9201 (OpenSearch)")
    parser.add_argument("--target-index", required=True)
    parser.add_argument("--dimension", type=int, default=648, help="signature length (648 for default n_grid=9)")
    parser.add_argument("--engine", default="lucene", choices=["lucene", "faiss", "nmslib"])
    parser.add_argument("--space-type", default="l2", help="l2, cosinesimil, innerproduct, hamming (engine-dependent)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--delete-source", action="store_true", help="delete the source index after successful verification")
    args = parser.parse_args(argv)

    # build the source client, auto-detecting the server type if requested
    if args.source_type == "auto":
        probe = make_client(args.source_url, "opensearch")
        args.source_type = "os" if detect_server_type(probe) == "opensearch" else "es"
        probe.close()
        print(f"auto-detected source type: {args.source_type}")

    source = make_client(args.source_url, "opensearch" if args.source_type == "os" else "elasticsearch")
    target = make_client(args.target_url, "opensearch")

    print(f"migrating {args.source_index} @ {args.source_url} -> {args.target_index} @ {args.target_url}")
    stats = migrate_index(
        source,
        target,
        args.source_index,
        args.target_index,
        dimension=args.dimension,
        engine=args.engine,
        space_type=args.space_type,
        batch_size=args.batch_size,
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
