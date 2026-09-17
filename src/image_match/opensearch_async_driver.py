from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

from .elasticsearch_async_driver import AsyncSignatureES
from .opensearch_driver import _parse_duration

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch
    from opensearchpy import AsyncOpenSearch


class AsyncSignatureOpenSearch(AsyncSignatureES):
    """Asynchronous OpenSearch driver for image-match.

    Shares the record format and query DSL with AsyncSignatureES. Differs
    only in the opensearch-py client API: index() takes a 'body' parameter,
    refresh is a params string, and 'timeout'/'size' go through params
    because opensearch-py reserves those kwargs.

    Requires the 'opensearch-async' extra:
        pip install image-match[opensearch-async]

    Examples:
        >>> from opensearchpy import AsyncOpenSearch
        >>> from image_match.opensearch_async_driver import AsyncSignatureOpenSearch
        >>> os_client = AsyncOpenSearch()
        >>> ses = AsyncSignatureOpenSearch(os_client)
        >>> await ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
        >>> await ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')

    """

    def __init__(
        self,
        es: AsyncOpenSearch,
        index: str = "images",
        timeout: str = "10s",
        size: int = 100,
        delete_duplicates_limit: int = 10000,
        minimum_should_match: int | str | None = None,
        use_filter_context: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Extra setup for OpenSearch

        Args:
            es (opensearchpy.AsyncOpenSearch): an async opensearch-py client instance
            index (Optional[string]): a name for the OpenSearch index (default 'images')
            timeout (Optional[int]): how long to wait on an OpenSearch query, in seconds (default 10)
            size (Optional[int]): maximum number of OpenSearch results (default 100)
            delete_duplicates_limit (Optional[int]): maximum number of duplicate candidates
                scanned per delete_duplicates call (default 10000)
            minimum_should_match (Optional[int | str]): require this many word
                clauses to match — e.g. 2 or "3<75%"; prunes low-overlap
                candidates on large indexes (default None = match any word)
            use_filter_context (Optional[bool]): wrap the word disjunction in a
                filter context so OpenSearch skips BM25 scoring entirely;
                faster on large indexes but the returned 'score' is constant
                (default False)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        """
        # the async OpenSearch client is API-identical to AsyncElasticsearch
        # for the operations used here; the cast is purely for type checkers
        super().__init__(
            cast("AsyncElasticsearch", es),
            index=index,
            timeout=timeout,
            size=size,
            delete_duplicates_limit=delete_duplicates_limit,
            minimum_should_match=minimum_should_match,
            use_filter_context=use_filter_context,
            *args,
            **kwargs,
        )

    @override
    async def _search(self, body: dict[str, Any]) -> Any:
        # opensearch-py reserves the 'timeout'/'request_timeout' params for the
        # HTTP request timeout, so the ES-style query-level timeout string can't
        # be sent; map it to a numeric request timeout instead
        return await cast("AsyncOpenSearch", self.es).search(
            index=self.index,
            body=body,
            params={"size": self.size, "request_timeout": _parse_duration(self.timeout)},
        )

    @override
    async def insert_single_record(self, rec: dict[str, Any], refresh_after: bool = False) -> None:
        """Insert an image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after inserting,
                making the record searchable immediately (default False)

        """
        rec["timestamp"] = datetime.now()
        await cast("AsyncOpenSearch", self.es).index(index=self.index, body=rec, params={"refresh": "true" if refresh_after else "false"})
