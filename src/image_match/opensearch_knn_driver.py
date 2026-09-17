from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import numpy as np

from .elasticsearch_driver import format_hits
from .opensearch_driver import SignatureOpenSearch, _search_params
from .signature_database_base import PreFilter, SignatureDatabaseBase

if TYPE_CHECKING:
    from opensearchpy import OpenSearch


def knn_index_body(dimension: int, engine: str = "lucene", space_type: str = "l2", data_type: str = "float") -> dict[str, Any]:
    """Build the settings/mappings body for a hybrid k-NN index.

    Only 'signature' is mapped explicitly (as knn_vector); all other fields
    rely on dynamic mapping, so 'simple_word_*' fields still index as long
    and the classic word-overlap drivers keep working on the same index.

    Args:
        dimension (int): signature vector length (648 for default n_grid=9)
        engine (Optional[str]): knn engine — 'lucene' (default, no native deps)
            or 'faiss'; 'nmslib' is OS2-only and rejected for new indexes on
            OpenSearch 3
        space_type (Optional[str]): vector space — 'l2' (default) or
            'cosinesimil'/'innerproduct'/'hamming' depending on engine
        data_type (Optional[str]): knn_vector data type — 'float' (default) or
            'byte' (4x smaller; signatures are int8 so 'byte' fits losslessly;
            requires OpenSearch >= 2.9)

    Returns:
        a dict suitable for client.indices.create(index=..., body=...)

    """
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "signature": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "data_type": data_type,
                    "method": {"name": "hnsw", "engine": engine, "space_type": space_type},
                }
            }
        },
    }


class SignatureOpenSearchKNN(SignatureOpenSearch):
    """OpenSearch k-NN driver for image-match.

    Stores the same record format as the word drivers, but searches with an
    approximate nearest-neighbour query (HNSW) directly on the 'signature'
    vector instead of the simple_word_* term disjunction. The k-NN hit list
    is rescored client-side with the usual normalized distance, so output
    format and distance_cutoff semantics are identical to SignatureES.

    Trade-off vs. the word drivers: HNSW is approximate — it may miss a
    candidate the word query would have found, and vice versa. 'size' acts
    as the candidate count (k); raise it for better recall.

    The index must be created with knn enabled — see knn_index_body() or
    tools/migrate_to_knn.py for converting an existing index.

    Requires the 'opensearch' extra:
        pip install image-match[opensearch]

    Examples:
        >>> from opensearchpy import OpenSearch
        >>> from image_match.opensearch_knn_driver import SignatureOpenSearchKNN, knn_index_body
        >>> os_client = OpenSearch()
        >>> os_client.indices.create(index="images_knn", body=knn_index_body(648))
        >>> ses = SignatureOpenSearchKNN(os_client, index="images_knn")
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
        self, es: OpenSearch, index: str = "images", timeout: str = "10s", size: int = 100, delete_duplicates_limit: int = 10000, *args: Any, **kwargs: Any
    ) -> None:
        """Extra setup for OpenSearch k-NN

        Args:
            es (opensearchpy.OpenSearch): an instance of the opensearch-py client
            index (Optional[string]): a name for the OpenSearch index (default 'images')
            timeout (Optional[int]): how long to wait on an OpenSearch query, in seconds (default 10)
            size (Optional[int]): number of ANN candidates (k) fetched per
                search before distance rescoring (default 100)
            delete_duplicates_limit (Optional[int]): maximum number of duplicate candidates
                scanned per delete_duplicates call (default 10000)
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        """
        # SignatureOpenSearch's ctor would bind positional *args to
        # minimum_should_match; call the base directly so they reach the
        # signature parameters (k, N, ...) as with the other drivers
        self.es = es
        self.index = index
        self.timeout = timeout
        self.size = size
        self.delete_duplicates_limit = delete_duplicates_limit

        SignatureDatabaseBase.__init__(self, *args, **kwargs)

    @override
    def search_single_record(self, rec: dict[str, Any], pre_filter: PreFilter = None) -> list[dict[str, Any]]:
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

        hits = self._search(body)["hits"]["hits"]

        return format_knn_hits(hits, np.array(signature), self.distance_cutoff)

    @override
    def _search(self, body: dict[str, Any]) -> Any:
        # the knn query carries its own size in the body (k candidates)
        return self.es.search(index=self.index, body=body, params=_search_params(self.timeout))


def build_knn_query(signature: Any, k: int, pre_filter: PreFilter = None) -> dict[str, Any]:
    """Build the knn query over the record's signature vector.

    Args:
        signature: the image signature (list or ndarray)
        k: number of approximate nearest neighbours to fetch
        pre_filter: an OpenSearch filter clause (dict) or list of clauses
            applied inside the knn query

    """
    knn: dict[str, Any] = {"vector": np.asarray(signature, dtype=float).tolist(), "k": k}

    if pre_filter is not None:
        if isinstance(pre_filter, list):
            knn["filter"] = {"bool": {"filter": pre_filter}}
        else:
            knn["filter"] = pre_filter

    return {"size": k, "query": {"knn": {"signature": knn}}, "_source": {"excludes": ["simple_word_*"]}}


def format_knn_hits(hits: list[dict[str, Any]], signature: np.ndarray, distance_cutoff: float) -> list[dict[str, Any]]:
    """Rescore knn candidates with normalized_distance and filter by cutoff.

    k-NN hits arrive in the same {_id, _score, _source} shape as word-query
    hits, so rescoring is shared with elasticsearch_driver.format_hits.
    """
    return format_hits(hits, signature, distance_cutoff)
