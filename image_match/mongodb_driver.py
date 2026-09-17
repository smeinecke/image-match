from multiprocessing import cpu_count
from queue import Empty, Queue
from threading import Thread

import numpy as np

from .signature_database_base import SignatureDatabaseBase, normalized_distance


class SignatureMongo(SignatureDatabaseBase):
    """MongoDB driver for image-match"""

    def __init__(self, collection, *args, **kwargs):
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
              'id': u'AVM37nMg0osmmAxpPvx7',
              'path': u'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg',
              'score': 0.28797293}
            ]

        """
        self.collection = collection
        self.index_names = []
        # Extract index fields, if any exist yet
        if self.collection.count_documents({}) > 0:
            self.index_names = [field for field in self.collection.find_one({}).keys() if field.find("simple") > -1]

        super(SignatureMongo, self).__init__(*args, **kwargs)

    def search_single_record(self, rec, pre_filter=None, *, n_parallel_words=1, word_limit=None, process_timeout=None, maximum_matches=1000):
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

        if word_limit is None:
            word_limit = self.N

        initial_q = Queue()

        for field_name in self.index_names[:word_limit]:
            initial_q.put({field_name: rec[field_name]})

        # enqueue a sentinel value so we know we have reached the end of the queue
        initial_q.put("STOP")
        queue_empty = False

        # create an empty queue for results
        results_q = Queue()

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
                    for key in results.keys():
                        if key not in unique_results:
                            unique_results.add(key)
                            match_list.append(results[key])

            for thread in p:
                thread.join()

            # yield a set of results
            if queue_empty:
                break

        return match_list

    def insert_single_record(self, rec):
        """Insert an image record, creating the word indexes if needed.

        Args:
            rec (dict): an image record in the format returned by make_record

        """
        self.collection.insert_one(rec)

        # if the collection has no indexes (except possibly '_id'), build them
        if len(self.collection.index_information()) <= 1:
            self.index_collection()

    def index_collection(self):
        """Index a collection on words."""
        # Index on words
        self.index_names = [field for field in self.collection.find_one({}).keys() if field.find("simple") > -1]
        for name in self.index_names:
            self.collection.create_index(name)


def get_next_match(result_q, word, collection, signature, cutoff=0.5, max_in_cursor=100, pre_filter=None):
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
    query = dict(word)
    if pre_filter:
        query.update(pre_filter)

    # if the query has many matches, then it's probably not a huge help. Get the next one.
    if collection.count_documents(query) > max_in_cursor:
        result_q.put("STOP")
        return

    curs = collection.find(query, projection=["_id", "signature", "path", "metadata"])

    matches = {}
    while True:
        try:
            rec = next(curs)
            dist = normalized_distance(np.reshape(signature, (1, signature.size)), np.array(rec["signature"]))[0]
            if dist < cutoff:
                matches[rec["_id"]] = {"dist": dist, "path": rec["path"], "id": rec["_id"], "metadata": rec.get("metadata")}
                result_q.put(matches)
        except StopIteration:
            # do nothing...the cursor is exhausted
            break
    result_q.put("STOP")
