# pyright: reportIncompatibleMethodOverride=false
# This module intentionally re-declares the synchronous driver API as
# coroutines — async variants are not LSP-substitutable by design.
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, override

import numpy as np

from .opensearch_driver import _parse_duration
from .opensearch_knn_driver import build_knn_query, format_knn_hits
from .signature_database_base import ImageInput, PreFilter, SignatureDatabaseBase, dedupe_results, make_record

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


class AsyncSignatureOpenSearchKNN(SignatureDatabaseBase):
    """Asynchronous OpenSearch k-NN driver for image-match.

    Async counterpart of SignatureOpenSearchKNN: every database-touching
    method is a coroutine, image decoding/signature generation is offloaded
    to a worker thread, and all_orientations searches run concurrently.
    Suited for embedding in async services such as FastAPI.

    The index must be created with knn enabled — see
    opensearch_knn_driver.knn_index_body() or tools/migrate_to_knn.py.

    Requires the 'opensearch-async' extra:
        pip install image-match[opensearch-async]

    Examples:
        >>> from opensearchpy import AsyncOpenSearch
        >>> from image_match.opensearch_knn_async_driver import AsyncSignatureOpenSearchKNN
        >>> os_client = AsyncOpenSearch()
        >>> ses = AsyncSignatureOpenSearchKNN(os_client, index="images_knn")
        >>> await ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
        >>> await ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')

    """

    def __init__(
        self, es: AsyncOpenSearch, index: str = "images", timeout: str = "10s", size: int = 100, delete_duplicates_limit: int = 10000, *args: Any, **kwargs: Any
    ) -> None:
        """Extra setup for OpenSearch k-NN

        Args:
            es (opensearchpy.AsyncOpenSearch): an async opensearch-py client instance
            index (Optional[string]): a name for the OpenSearch index (default 'images')
            timeout (Optional[int]): how long to wait on an OpenSearch query, in seconds (default 10)
            size (Optional[int]): number of ANN candidates (k) fetched per
                search before distance rescoring (default 100)
            delete_duplicates_limit (Optional[int]): maximum number of duplicate candidates
                scanned per delete_duplicates call (default 10000)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        """
        self.es = es
        self.index = index
        self.timeout = timeout
        self.size = size
        self.delete_duplicates_limit = delete_duplicates_limit

        super().__init__(*args, **kwargs)

    async def _search(self, body: dict[str, Any]) -> Any:
        return await self.es.search(index=self.index, body=body, params={"request_timeout": _parse_duration(self.timeout)})

    @override
    async def search_single_record(self, rec: dict[str, Any], pre_filter: PreFilter = None) -> list[dict[str, Any]]:
        """Search for a matching image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            pre_filter (Optional[dict]): an OpenSearch filter clause applied
                inside the knn query (default None)

        Returns:
            a list of dicts representing matches, filtered by distance_cutoff

        """
        signature = rec.pop("signature")
        body = build_knn_query(signature, self.size, pre_filter)

        hits = (await self._search(body))["hits"]["hits"]

        return format_knn_hits(hits, np.array(signature), self.distance_cutoff)

    @override
    async def insert_single_record(self, rec: dict[str, Any], refresh_after: bool = False) -> None:
        """Insert an image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after inserting,
                making the record searchable immediately (default False)

        """
        rec["timestamp"] = datetime.now()
        await self.es.index(index=self.index, body=rec, params={"refresh": "true" if refresh_after else "false"})

    @override
    async def add_image(
        self, path: str, img: ImageInput | None = None, bytestream: bool = False, metadata: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Add a single image to the database.

        Signature generation runs in a worker thread; only the index request
        is awaited.
        """
        rec = await asyncio.to_thread(make_record, path, self.gis, self.k, self.N, img=img, bytestream=bytestream, metadata=metadata)
        await self.insert_single_record(rec, *args, **kwargs)

    @override
    async def search_image(
        self, path: ImageInput, all_orientations: bool = False, bytestream: bool = False, pre_filter: PreFilter = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Search for matches.

        Image preprocessing and signature generation run in a worker thread;
        the per-orientation knn searches are awaited concurrently.
        """
        records = await asyncio.to_thread(self._orientation_records, path, all_orientations, bytestream)

        # run all orientation queries concurrently — with all_orientations=True
        # this collapses 16 sequential round trips into one batch
        per_record = await asyncio.gather(*(self.search_single_record(rec, pre_filter=pre_filter, **kwargs) for rec in records))

        return dedupe_results([match for matches in per_record for match in matches])

    async def delete_duplicates(self, path: str, limit: int | None = None) -> None:
        """Delete all but one entries in the index whose `path` value is equivalent to that of path.

        Args:
            path (string): path value to compare to those in the index
            limit (Optional[int]): maximum number of duplicate candidates to scan;
                defaults to the instance's delete_duplicates_limit (default 10000)

        """
        if limit is None:
            limit = self.delete_duplicates_limit

        matching_paths = [item["_id"] for item in await self._path_hits(path, limit) if item["_source"].get("path") == path]

        for id_tag in matching_paths[1:]:
            await self.es.delete(index=self.index, id=id_tag)

    async def _path_hits(self, path: str, limit: int) -> list[dict[str, Any]]:
        """Search for documents whose path field fuzzy-matches `path`."""
        return (await self.es.search(body={"query": {"match": {"path": path}}}, index=self.index, params={"size": limit}))["hits"]["hits"]
