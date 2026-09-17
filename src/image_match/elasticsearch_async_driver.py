# pyright: reportIncompatibleMethodOverride=false
# This module intentionally re-declares the synchronous driver API as
# coroutines — async variants are not LSP-substitutable by design.
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, override

from .elasticsearch_driver import SearchClient, build_word_query, duplicate_ids, exact_path_ids, format_hits, helpers_module
from .signature_database_base import ImageInput, PreFilter, SignatureDatabaseBase, _normalize_metadata, dedupe_results, make_record


class AsyncSignatureES[ClientT: SearchClient](SignatureDatabaseBase):
    """Asynchronous Elasticsearch driver for image-match.

    API-identical to SignatureES, but every database-touching method is a
    coroutine and image decoding/signature generation is offloaded to a
    worker thread so the event loop is never blocked. Suited for embedding
    in async services such as FastAPI.

    Requires the 'elasticsearch-async' extra:
        pip install image-match[elasticsearch-async]

    Examples:
        >>> from elasticsearch import AsyncElasticsearch
        >>> from image_match.elasticsearch_async_driver import AsyncSignatureES
        >>> es = AsyncElasticsearch()
        >>> ses = AsyncSignatureES(es)
        >>> await ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
        >>> await ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
        [
         {'dist': 0.0,
          'id': 'AVM37nMg0osmmAxpPvx6',
          'path': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
          'score': 0.28797293}
        ]

    """

    def __init__(
        self,
        es: ClientT,
        index: str = "images",
        timeout: str = "10s",
        size: int = 100,
        delete_duplicates_limit: int = 10000,
        minimum_should_match: int | str | None = None,
        use_filter_context: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Extra setup for Elasticsearch

        Args:
            es (elasticsearch.AsyncElasticsearch): an async elasticsearch client instance
            index (Optional[string]): a name for the Elasticsearch index (default 'images')
            timeout (Optional[int]): how long to wait on an Elasticsearch query, in seconds (default 10)
            size (Optional[int]): maximum number of Elasticsearch results (default 100)
            delete_duplicates_limit (Optional[int]): maximum number of duplicate candidates
                scanned per delete_duplicates call (default 10000)
            minimum_should_match (Optional[int | str]): require this many word
                clauses to match — e.g. 2 or "3<75%"; prunes low-overlap
                candidates on large indexes (default None = match any word)
            use_filter_context (Optional[bool]): wrap the word disjunction in a
                filter context so Elasticsearch skips BM25 scoring entirely;
                faster on large indexes but the returned 'score' is constant
                (default False)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        """
        self.es = es
        self.index = index
        self.timeout = timeout
        self.size = size
        self.delete_duplicates_limit = delete_duplicates_limit
        self.minimum_should_match = minimum_should_match
        self.use_filter_context = use_filter_context

        super().__init__(*args, **kwargs)

    async def _search(self, body: dict[str, Any]) -> Any:
        """Run the word-match search (async counterpart of SignatureES._search)."""
        return await self.es.search(index=self.index, body=body, size=self.size, timeout=self.timeout)

    @override
    async def search_single_record(self, rec: dict[str, Any], pre_filter: PreFilter = None) -> list[dict[str, Any]]:
        """Search for a matching image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            pre_filter (Optional[dict]): an Elasticsearch filter clause applied
                before matching (default None)

        Returns:
            a list of dicts representing matches, filtered by distance_cutoff

        """
        body = build_word_query(rec, pre_filter, self.minimum_should_match, self.use_filter_context)
        signature = rec.pop("signature")

        res = (await self._search(body))["hits"]["hits"]

        return format_hits(res, signature, self.distance_cutoff)

    @override
    async def insert_single_record(self, rec: dict[str, Any], refresh_after: bool = False) -> None:
        """Insert an image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after inserting,
                making the record searchable immediately (default False)

        """
        rec["timestamp"] = datetime.now()
        await self.es.index(index=self.index, document=rec, refresh=refresh_after)

    @override
    async def add_image(
        self, path: str, img: ImageInput | None = None, bytestream: bool = False, metadata: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Add a single image to the database.

        Signature generation runs in a worker thread; only the index request
        is awaited.

        Args:
            path (string): path or identifier for image. If img=None, then path is assumed to be
                a URL or filesystem path
            img (Optional[string]): usually raw image data. In this case, path will still be stored, but
                a signature will be generated from data in img (default None)
            bytestream (Optional[boolean]): will the image be passed as raw bytes? (default False)
            metadata (Optional): any other information you want to include, can be nested (default None)
            *args: Variable length argument list to pass to insert_single_record
            **kwargs: Arbitrary keyword arguments to pass to insert_single_record

        """
        rec = await asyncio.to_thread(make_record, path, self.gis, self.k, self.N, img=img, bytestream=bytestream, metadata=metadata)
        await self.insert_single_record(rec, *args, **kwargs)

    @override
    async def add_images(
        self,
        paths: list[ImageInput],
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        bytestream: bool = False,
        n_threads: int = 1,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Generate signatures for many images and bulk-insert the records.

        Signature generation runs in a worker thread (with its own thread pool
        when n_threads > 1); only the bulk request is awaited.

        Args:
            paths (list): paths, URLs, or raw image data
            metadata (Optional): a single dict applied to every record, or a
                list of dicts/None aligned with paths (default None)
            bytestream (Optional[boolean]): inputs are raw image bytes (default False)
            n_threads (Optional[int]): signature-generation threads (default 1)
            chunk_size (Optional[int]): process and insert in batches of this many
                images instead of buffering all records in memory (default None =
                one bulk request)
            **kwargs: passed to insert_records (e.g. refresh_after — honored on
                the final chunk only)

        Returns:
            the number of successfully indexed records

        """
        paths = list(paths)
        metas = _normalize_metadata(metadata, len(paths))

        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer (got {chunk_size})")

        if not chunk_size or chunk_size >= len(paths):
            records = await asyncio.to_thread(self._make_records, paths, metas, bytestream, n_threads)
            return await self.insert_records(records, **kwargs)

        total = 0
        for start in range(0, len(paths), chunk_size):
            records = await asyncio.to_thread(self._make_records, paths[start : start + chunk_size], metas[start : start + chunk_size], bytestream, n_threads)
            chunk_kwargs = dict(kwargs)
            if start + chunk_size < len(paths):
                chunk_kwargs["refresh_after"] = False  # refresh once, on the last chunk
            total += await self.insert_records(records, **chunk_kwargs)
        return total

    @override
    async def insert_records(self, records: list[dict[str, Any]], refresh_after: bool = False, **kwargs: Any) -> int:
        """Bulk-insert pre-made records (see add_images).

        Args:
            records (list[dict]): image records in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after the bulk
                request, making records searchable immediately (default False)
            **kwargs: extra keyword arguments forwarded to helpers.async_bulk

        Returns:
            the number of successfully indexed records

        """
        for rec in records:
            rec["timestamp"] = datetime.now()
        ok, _ = await helpers_module(self.es).async_bulk(self.es, [{"_index": self.index, "_source": rec} for rec in records], refresh=refresh_after, **kwargs)
        return ok

    @override
    async def search_image(
        self, path: ImageInput, all_orientations: bool = False, bytestream: bool = False, pre_filter: PreFilter = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search for matches.

        Image preprocessing and signature generation run in a worker thread;
        the search requests are awaited.

        Args:
            path (string): path or image data. If bytestream=False, then path is assumed to be
                a URL or filesystem path. Otherwise, it's assumed to be raw image data
            all_orientations (Optional[boolean]): if True, search for all combinations of mirror
                images, rotations, and color inversions (default False)
            bytestream (Optional[boolean]): will the image be passed as raw bytes? (default False)
            pre_filter (Optional[dict]): filters list before applying the matching algorithm
                (default None)
            **kwargs: Arbitrary keyword arguments to pass to search_single_record

        Returns:
            a formatted list of dicts representing unique matches, sorted by dist

        """
        records = await asyncio.to_thread(self._orientation_records, path, all_orientations, bytestream)

        # run all orientation queries concurrently — with all_orientations=True
        # this collapses 16 sequential round trips into one batch
        per_record = await asyncio.gather(*(self.search_single_record(rec, pre_filter=pre_filter, **kwargs) for rec in records))

        return dedupe_results([match for matches in per_record for match in matches])

    async def delete_duplicates(self, path: str, limit: int | None = None) -> None:
        """Delete all but one entries in elasticsearch whose `path` value is equivalent to that of path.

        Args:
            path (string): path value to compare to those in the elastic search
            limit (Optional[int]): maximum number of duplicate candidates to scan;
                defaults to the instance's delete_duplicates_limit (default 10000)

        """
        if limit is None:
            limit = self.delete_duplicates_limit

        for id_tag in duplicate_ids(await self._path_hits(path, limit), path):
            await self.es.delete(index=self.index, id=id_tag)

    async def delete_image(self, path: str, limit: int | None = None) -> int:
        """Delete all records whose stored path exactly equals `path`.

        The path match query is fuzzy, so hits are filtered to exact matches
        before deletion — documents with merely similar paths are kept.

        Args:
            path (string): path value to remove entirely from the index
            limit (Optional[int]): maximum number of candidates to scan;
                defaults to the instance's delete_duplicates_limit (default 10000)

        Returns:
            the number of documents deleted

        """
        if limit is None:
            limit = self.delete_duplicates_limit

        ids = exact_path_ids(await self._path_hits(path, limit), path)
        for id_tag in ids:
            await self.es.delete(index=self.index, id=id_tag)
        return len(ids)

    async def _path_hits(self, path: str, limit: int) -> list[dict[str, Any]]:
        """Search for documents whose path field fuzzy-matches `path`."""
        return (await self.es.search(body={"query": {"match": {"path": path}}}, index=self.index, size=limit))["hits"]["hits"]
