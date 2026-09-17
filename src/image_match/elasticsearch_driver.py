from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, override

import numpy as np

from .signature_database_base import PreFilter, SignatureDatabaseBase, normalized_distance

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


class SignatureES(SignatureDatabaseBase):
    """Elasticsearch driver for image-match"""

    def __init__(
        self,
        es: Elasticsearch,
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
            es (elasticsearch): an instance of the elasticsearch python driver
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

        Examples:
            >>> from elasticsearch import Elasticsearch
            >>> from image_match.elasticsearch_driver import SignatureES
            >>> es = Elasticsearch()
            >>> ses = SignatureES(es)
            >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            [
             {'dist': 0.0,
              'id': 'AVM37nMg0osmmAxpPvx6',
              'path': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.es = es
        self.index = index
        self.timeout = timeout
        self.size = size
        self.delete_duplicates_limit = delete_duplicates_limit
        self.minimum_should_match = minimum_should_match
        self.use_filter_context = use_filter_context

        super().__init__(*args, **kwargs)

    @override
    def search_single_record(self, rec: dict[str, Any], pre_filter: PreFilter = None) -> list[dict[str, Any]]:
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

        res = self._search(body)["hits"]["hits"]

        return format_hits(res, signature, self.distance_cutoff)

    def _search(self, body: dict[str, Any]) -> Any:
        """Run the word-match search.

        Isolated so subclasses can adapt the call to clients with different
        signatures (e.g. opensearch-py).
        """
        return self.es.search(index=self.index, body=body, size=self.size, timeout=self.timeout)

    @override
    def insert_single_record(self, rec: dict[str, Any], refresh_after: bool = False) -> None:
        """Insert an image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            refresh_after (Optional[boolean]): refresh the index after inserting,
                making the record searchable immediately (default False)

        """
        rec["timestamp"] = datetime.now()
        self.es.index(index=self.index, document=rec, refresh=refresh_after)

    def delete_duplicates(self, path: str, limit: int | None = None) -> None:
        """Delete all but one entries in elasticsearch whose `path` value is equivalent to that of path.

        Args:
            path (string): path value to compare to those in the elastic search
            limit (Optional[int]): maximum number of duplicate candidates to scan;
                defaults to the instance's delete_duplicates_limit (default 10000)

        """
        if limit is None:
            limit = self.delete_duplicates_limit

        matching_paths = [item["_id"] for item in self._path_hits(path, limit) if item["_source"].get("path") == path]

        for id_tag in matching_paths[1:]:
            self.es.delete(index=self.index, id=id_tag)

    def _path_hits(self, path: str, limit: int) -> list[dict[str, Any]]:
        """Search for documents whose path field fuzzy-matches `path`."""
        return self.es.search(body={"query": {"match": {"path": path}}}, index=self.index, size=limit)["hits"]["hits"]


def build_word_query(
    rec: dict[str, Any],
    pre_filter: PreFilter = None,
    minimum_should_match: int | str | None = None,
    use_filter_context: bool = False,
) -> dict[str, Any]:
    """Build the bool/should term query over a record's simple_word_* fields.

    Removes 'path' and 'metadata' from rec; the caller pops 'signature'.
    Stray stored fields (e.g. timestamp) must not leak into the query.

    Args:
        rec: an image record in the format returned by make_record
        pre_filter: an ES filter clause (or list of clauses) applied before matching
        minimum_should_match: how many word clauses must match (default None = 1)
        use_filter_context: evaluate the word disjunction in filter context,
            skipping scoring entirely (the returned _score is then constant)

    """
    rec.pop("path", None)
    rec.pop("metadata", None)

    should = [{"term": {word: rec[word]}} for word in rec if word.startswith("simple_word_")]
    body: dict[str, Any] = {"_source": {"excludes": ["simple_word_*"]}}

    if use_filter_context:
        word_bool: dict[str, Any] = {"should": should, "minimum_should_match": minimum_should_match or 1}
        filters = [pre_filter] if isinstance(pre_filter, dict) else list(pre_filter or [])
        filters.append({"bool": word_bool})
        body["query"] = {"bool": {"filter": filters}}
    else:
        word_bool = {"should": should}
        if minimum_should_match is not None:
            word_bool["minimum_should_match"] = minimum_should_match
        if pre_filter is not None:
            word_bool["filter"] = pre_filter
        body["query"] = {"bool": word_bool}

    return body


def format_hits(hits: list[dict[str, Any]], signature: np.ndarray, distance_cutoff: float) -> list[dict[str, Any]]:
    """Compute distances to a signature and return cutoff-filtered hit dicts."""
    sigs = np.array([x["_source"]["signature"] for x in hits])

    if sigs.size == 0:
        return []

    dists = normalized_distance(sigs, np.array(signature))

    formatted_res = [
        {"id": x["_id"], "score": x["_score"], "metadata": x["_source"].get("metadata"), "path": x["_source"].get("url", x["_source"].get("path"))}
        for x in hits
    ]

    for i, row in enumerate(formatted_res):
        row["dist"] = dists[i]
    return [y for y in formatted_res if y["dist"] < distance_cutoff]
