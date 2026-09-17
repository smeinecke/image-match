from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

from .elasticsearch_driver import SignatureES

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from opensearchpy import OpenSearch


def _parse_duration(timeout: str) -> float:
    """Parse an ES-style duration string ('10s', '500ms', '1m') to seconds."""
    s = str(timeout).strip()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    return float(s)


class SignatureOpenSearch(SignatureES):
    """OpenSearch driver for image-match.

    Shares the record format and query DSL with SignatureES -- OpenSearch is
    API-compatible with Elasticsearch 7.x for the operations used here. The
    only differences are in the opensearch-py client API: index() still takes
    a 'body' parameter, refresh is a params string, and 'timeout'/'size' are
    passed through params because opensearch-py reserves those kwargs.

    Install the client with the 'opensearch' extra:
        pip install image-match[opensearch]

    Examples:
        >>> from opensearchpy import OpenSearch
        >>> from image_match.opensearch_driver import SignatureOpenSearch
        >>> os_client = OpenSearch()
        >>> ses = SignatureOpenSearch(os_client)
        >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
        >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
        [
         {'dist': 0.0,
          'id': 'AVM37nMg0osmmAxpPvx6',
          'path': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
          'score': 0.28797293}
        ]

    """

    def __init__(
        self,
        es: OpenSearch,
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
            es (opensearchpy.OpenSearch): an instance of the opensearch-py client
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
        # the OpenSearch client is API-identical to the Elasticsearch client
        # for the operations used here; the cast is purely for type checkers
        super().__init__(
            cast("Elasticsearch", es),
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
    def _search(self, body: dict[str, Any]) -> Any:
        # opensearch-py reserves the 'timeout'/'request_timeout' params for the
        # HTTP request timeout, so the ES-style query-level timeout string can't
        # be sent; map it to a numeric request timeout instead
        return cast("OpenSearch", self.es).search(
            index=self.index,
            body=body,
            params={"size": self.size, "request_timeout": _parse_duration(self.timeout)},
        )

    @override
    def insert_single_record(self, rec: dict[str, Any], refresh_after: bool = False) -> None:
        """Insert an image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after inserting,
                making the record searchable immediately (default False)

        """
        rec["timestamp"] = datetime.now()
        cast("OpenSearch", self.es).index(index=self.index, body=rec, params={"refresh": "true" if refresh_after else "false"})
