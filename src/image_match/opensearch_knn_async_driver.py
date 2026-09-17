# pyright: reportIncompatibleMethodOverride=false
# This module intentionally re-declares the synchronous driver API as
# coroutines — async variants are not LSP-substitutable by design.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from .opensearch_async_driver import AsyncSignatureOpenSearch
from .opensearch_driver import _search_params
from .opensearch_knn_driver import build_knn_query, format_knn_hits
from .signature_database_base import PreFilter, SignatureDatabaseBase

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


class AsyncSignatureOpenSearchKNN(AsyncSignatureOpenSearch):
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
        # AsyncSignatureOpenSearch's ctor would bind positional *args to
        # minimum_should_match; call the base directly so they reach the
        # signature parameters (k, N, ...) as with the other drivers
        self.es = es
        self.index = index
        self.timeout = timeout
        self.size = size
        self.delete_duplicates_limit = delete_duplicates_limit

        SignatureDatabaseBase.__init__(self, *args, **kwargs)

    @override
    async def _search(self, body: dict[str, Any]) -> Any:
        # the knn query carries its own size in the body (k candidates)
        return await self.es.search(index=self.index, body=body, params=_search_params(self.timeout))

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
