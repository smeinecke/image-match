from __future__ import annotations

from multiprocessing import cpu_count
from queue import Empty, Queue
from threading import Thread
from typing import TYPE_CHECKING, Any, override

import numpy as np

from .signature_database_base import PreFilter, SignatureDatabaseBase, normalized_distance

if TYPE_CHECKING:
    from pymongo.collection import Collection


class SignatureMongo(SignatureDatabaseBase):
    """MongoDB driver for image-match"""

    def __init__(self, collection: Collection, *args: Any, **kwargs: Any) -> None:
        """Additional MongoDB setup

        Args:
            collection (collection): a MongoDB collection instance
            args (Optional): Variable length argument list to pass to base constructor
            kwargs (Optional): Arbitrary keyword arguments to pass to base constructor

        Examples:
            >>> from image_match.mongodb_driver import SignatureMongo
            >>> from pymongo import MongoClient
            >>> client = MongoClient(connect=False)
            >>> c = client.images.images
            >>> ses = SignatureMongo(c)
            >>> ses.add_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            >>> ses.search_image('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
            [
             {'dist': 0.0,
              'id': 'AVM37nMg0osmmAxpPvx7',
              'path': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.collection = collection
        self.index_names: list[str] = []
        # Extract index fields, if any exist yet
        if self.collection.count_documents({}) > 0:
            self._load_index_names()

        super().__init__(*args, **kwargs)

    @override
    def search_single_record(
        self,
        rec: dict,
        pre_filter: PreFilter = None,
        *,
        n_parallel_words: int | None = 1,
        word_limit: int | None = None,
        process_timeout: float | None = None,
        maximum_matches: int = 1000,
    ) -> list[dict]:
        """Search for a matching image record.

        Args:
            rec (dict): an image record in the format returned by make_record
            pre_filter (Optional[dict]): additional query conditions merged into
                each find query (default None)
            n_parallel_words (Optional[int]): number of words to scan in parallel;
                if None, uses cpu_count() (default 1)
            word_limit (Optional[int]): only scan this many words; if None, all
                N words are used (default None)
            process_timeout (Optional[float]): seconds to wait for a worker result
                before giving up; None waits indefinitely (default None)
            maximum_matches (Optional[int]): if a word matches more records than
                this, the word is considered non-discriminatory and skipped
                (default 1000)

        Returns:
            a list of dicts representing unique matches

        """
        if n_parallel_words is None:
            n_parallel_words = cpu_count()
        if n_parallel_words < 1:
            raise ValueError("n_parallel_words must be at least 1")

        if word_limit is None:
            word_limit = self.N

        initial_q = self._word_query_queue(rec, word_limit)
        queue_empty = False

        # create an empty queue for results
        results_q: Queue = Queue()

        # create a set of unique results, using MongoDB _id field
        unique_results = set()

        match_list = []

        while True:
            # build children threads, taking cursors from in_process queue first, then initial queue
            p = []
            while len(p) < n_parallel_words:
                word_pair = initial_q.get()
                if word_pair == "STOP":
                    # if we reach the sentinel value, set the flag and stop queuing threads
                    queue_empty = True
                    break
                p.append(
                    Thread(
                        target=get_next_match,
                        daemon=True,
                        args=(results_q, word_pair, self.collection, np.array(rec["signature"]), self.distance_cutoff, maximum_matches, pre_filter),
                    )
                )

            if not p:
                break

            for thread in p:
                thread.start()
            # collect results, taking care not to return the same result twice

            num_workers = len(p)

            while num_workers:
                try:
                    results = results_q.get(timeout=process_timeout)
                except Empty:
                    num_workers = 0
                    break
                if results == "STOP":
                    num_workers -= 1
                else:
                    for key in results:
                        if key not in unique_results:
                            unique_results.add(key)
                            match_list.append(results[key])

            for thread in p:
                # bound the wait so a timed-out search doesn't block on stragglers;
                # workers are daemon threads and their late results are discarded
                thread.join(timeout=process_timeout)

            # yield a set of results
            if queue_empty:
                break

        return match_list

    def _word_query_queue(self, rec: dict, word_limit: int) -> Queue:
        """Build the work queue of {word_field: word_value} lookups.

        Lazily repopulates index_names if the collection was populated after
        this instance was created, and ends with a 'STOP' sentinel.

        Args:
            rec (dict): an image record in the format returned by make_record
            word_limit (int): only use this many word fields

        Returns:
            a Queue of single-entry dicts, terminated by 'STOP'

        """
        # index_names may be empty if the collection was populated after
        # this instance was created; try to pick the fields up lazily
        if not self.index_names:
            self._load_index_names()

        initial_q: Queue = Queue()
        for field_name in self.index_names[:word_limit]:
            # a field may be absent from rec if the index was built with a
            # different N (number of words); skip instead of raising KeyError
            if field_name in rec:
                initial_q.put({field_name: rec[field_name]})

        # enqueue a sentinel value so we know we have reached the end of the queue
        initial_q.put("STOP")
        return initial_q

    @override
    def insert_single_record(self, rec: dict) -> None:
        """Insert an image record, creating the word indexes if needed.

        Args:
            rec (dict): an image record in the format returned by make_record

        """
        self.collection.insert_one(rec)

        # if the collection has no indexes (except possibly '_id'), build them
        if len(self.collection.index_information()) <= 1:
            self.index_collection()

    def index_collection(self) -> None:
        """Index a collection on words."""
        self._load_index_names()
        for name in self.index_names:
            self.collection.create_index(name)

    def _load_index_names(self) -> None:
        """(Re)read the simple-word field names from a sample document."""
        doc = self.collection.find_one({}) or {}
        self.index_names = [field for field in doc if "simple" in field]


def get_next_match(
    result_q: Queue, word: dict, collection: Collection, signature: np.ndarray, cutoff: float = 0.5, max_in_cursor: int = 100, pre_filter: PreFilter = None
) -> None:
    """Given a cursor, iterate through matches

    Scans a cursor for word matches below a distance threshold.
    Exhausts a cursor, possibly enqueuing many matches
    Note that placing this function outside the SignatureCollection
    class breaks encapsulation.  This is done for compatibility with
    multiprocessing.

    Args:
        result_q (queue.Queue): a queue in which to queue results
        word (dict): {word_name: word_value} dict to scan against
        collection (collection): a pymongo collection
        signature (numpy.ndarray): signature array to match against
        cutoff (Optional[float]): normalized distance limit (default 0.5)
        max_in_cursor (Optional[int]): if more than max_in_cursor matches are in the cursor,
            ignore this cursor; this column is not discriminatory (default 100)
        pre_filter (Optional[dict]): additional query conditions merged into the
            find query (default None)

    """
    try:
        query = dict(word)
        if pre_filter:
            query.update(pre_filter)

        # if the query has many matches, then it's probably not a huge help. Get the next one.
        if collection.count_documents(query) <= max_in_cursor:
            curs = collection.find(query, projection=["_id", "signature", "path", "metadata"])
            while True:
                try:
                    rec = next(curs)
                except StopIteration:
                    # do nothing...the cursor is exhausted
                    break
                dist = normalized_distance(np.reshape(signature, (1, signature.size)), np.array(rec["signature"]))[0]
                if dist < cutoff:
                    # put a fresh dict per match; sharing a growing dict across
                    # the queue races with the consumer iterating it
                    result_q.put({rec["_id"]: {"dist": dist, "path": rec.get("path"), "id": rec["_id"], "metadata": rec.get("metadata")}})
    finally:
        # always signal completion, even on error, so the consumer doesn't hang
        result_q.put("STOP")
