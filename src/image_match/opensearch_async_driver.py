from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from .elasticsearch_async_driver import AsyncSignatureES
from .opensearch_driver import _index_kwargs, _search_params

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


class AsyncSignatureOpenSearch(AsyncSignatureES["AsyncOpenSearch"]):
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
        super().__init__(
            es,
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
        return await self.es.search(
            index=self.index,
            body=body,
            params=_search_params(self.timeout, self.size),
        )

    @override
    async def insert_single_record(self, rec: dict[str, Any], refresh_after: bool = False) -> None:
        """Insert an image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after inserting,
                making the record searchable immediately (default False)

        """
        await self.es.index(**_index_kwargs(self.index, rec, refresh_after))
