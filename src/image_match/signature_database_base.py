from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
from operator import itemgetter
from typing import Any

import numpy as np

from .goldberg import ImageInput, ImageSignature

# a database filter clause: dict for MongoDB, dict or list of clauses for
# Elasticsearch (e.g. {"term": {"metadata.tenant_id": "foo"}})
type PreFilter = dict[str, Any] | list[dict[str, Any]] | None


class SignatureDatabaseBase:
    """Base class for storing and searching image signatures in a database

    Note:
        You must implement the methods search_single_record and insert_single_record
        in a derived class

    """

    def search_single_record(self, rec: dict[str, Any], pre_filter: PreFilter = None) -> list[dict[str, Any]]:
        """Search for a matching image record.

        Must be implemented by derived class.

        Args:
            rec (dict): an image record. Will be in the format returned by
                make_record

                For example, rec could have the form::

                    {'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
                     'signature': [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0 ... ]
                     'simple_word_0': 42252475,
                     'simple_word_1': 23885671,
                     'simple_word_10': 9967839,
                     'simple_word_11': 4257902,
                     'simple_word_12': 28651959,
                     'simple_word_13': 33773597,
                     'simple_word_14': 39331441,
                     'simple_word_15': 39327300,
                     'simple_word_16': 11337345,
                     'simple_word_17': 9571961,
                     'simple_word_18': 28697868,
                     'simple_word_19': 14834907,
                     'simple_word_2': 7434746,
                     'simple_word_20': 37985525,
                     'simple_word_21': 10753207,
                     'simple_word_22': 9566120,
                     ...
                     'metadata': {'category': 'art'},
                     }

                The number of simple words corresponds to the attribute N

            pre_filter (dict): a filter to be applied by the concrete implementation
                   before applying the matching strategy

                For example:
                    { "term": {  "image.metadata.category": "art" } }

        Returns:
            a formatted list of dicts representing matches.

            For example, if three matches are found::

                [
                 {'dist': 0.069116439263706961,
                  'id': 'AVM37oZq0osmmAxpPvx7',
                  'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg'},
                 {'dist': 0.22484320805049718,
                  'id': 'AVM37nMg0osmmAxpPvx6',
                  'path': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'},
                 {'dist': 0.42529792112113302,
                  'id': 'AVM37p530osmmAxpPvx9',
                  'metadata': {...},
                  'path': 'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
                ]

            You can return any fields you like, but must include at least dist and id. Duplicate entries are ok,
            and they do not need to be sorted

        """
        raise NotImplementedError

    def insert_single_record(self, rec: dict[str, Any]) -> None:
        """Insert an image record.

        Must be implemented by derived class.

        Args:
            rec (dict): an image record. Will be in the format returned by
                make_record

                For example, rec could have the form::

                    {'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
                     'signature': [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0 ... ]
                     'simple_word_0': 42252475,
                     'simple_word_1': 23885671,
                     'simple_word_10': 9967839,
                     'simple_word_11': 4257902,
                     'simple_word_12': 28651959,
                     'simple_word_13': 33773597,
                     'simple_word_14': 39331441,
                     'simple_word_15': 39327300,
                     'simple_word_16': 11337345,
                     'simple_word_17': 9571961,
                     'simple_word_18': 28697868,
                     'simple_word_19': 14834907,
                     'simple_word_2': 7434746,
                     'simple_word_20': 37985525,
                     'simple_word_21': 10753207,
                     'simple_word_22': 9566120,
                     ...
                     'metadata': {...}
                     }

                The number of simple words corresponds to the attribute N

        """
        raise NotImplementedError

    def __init__(
        self,
        k: int = 16,
        N: int = 63,
        n_grid: int = 9,
        crop_percentile: tuple[int, int] | None = (5, 95),
        distance_cutoff: float = 0.45,
        *signature_args: Any,
        **signature_kwargs: Any,
    ) -> None:
        """Set up storage scheme for images

        Central to the speed of this approach is the transforming the image
        signature into something that can be speedily indexed and matched.
        In our case, that means splitting the image signature into N words
        of length k, then encoding those words as integers. The idea here is
        that integer indices are more efficient than array indices.

        For example, say your image signature is [0, 1, 2, 0, -1, -2, 0, 1] and
        k=3 and N=4. That means we want 4 words of length 3.  For this signa-
        ture, that gives us:

        [0, 1, 2]
        [2, 0, -1]
        [-1, -2, 0]
        [0, 1]

        Note that signature elements can be repeated, and any mismatch in length
        is chopped off in the last word (which will be padded with zeros). Since
        these numbers run from -2..2, there 5 possibilites.  Adding 2 to each word
        makes them strictly non-negative, then the quantity, and transforming to
        base-5 makes unique integers. For the first word:

        [0, 1, 2] + 2 = [2, 3, 4]
        [5**0, 5**1, 5**2] = [1, 5, 25]
        dot([2, 3, 4], [1, 5, 25]) = 2 + 15 + 100 = 117

        So the integer word is 117.  Storing all the integer words as different
        database columns or fields gives us the speedy lookup. In practice, word
        arrays are 'squeezed' to between -1..1 before encoding.

        Args:
            k (Optional[int]): the width of a word (default 16)
            N (Optional[int]): the number of words (default 63)
            n_grid (Optional[int]): the n_grid x n_grid size to use in determining
                the image signature (default 9)
            crop_percentile (Optional[Tuple[int]]): lower and upper bounds when
                considering how much variance to keep in the image (default (5, 95))
            distance_cutoff (Optional [float]): maximum image signature distance to
                be considered a match (default 0.45)
            *signature_args: Variable length argument list to pass to ImageSignature
            **signature_kwargs: Arbitrary keyword arguments to pass to ImageSignature

        """
        # Check integer inputs
        if type(k) is not int:
            raise TypeError("k should be an integer")
        if type(N) is not int:
            raise TypeError("N should be an integer")
        if type(n_grid) is not int:
            raise TypeError("n_grid should be an integer")

        self.k = k
        self.N = N
        self.n_grid = n_grid

        # Check float input (runtime guard for untyped callers)
        if not isinstance(distance_cutoff, (int, float)):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("distance_cutoff should be a float")
        if distance_cutoff < 0.0:
            raise ValueError(f"distance_cutoff should be > 0 (got {distance_cutoff!r})")

        self.distance_cutoff = float(distance_cutoff)

        self.crop_percentile = crop_percentile

        self.gis = ImageSignature(n=n_grid, crop_percentiles=crop_percentile, *signature_args, **signature_kwargs)

    def add_image(
        self, path: str, img: ImageInput | None = None, bytestream: bool = False, metadata: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Add a single image to the database

        Args:
            path (string): path or identifier for image. If img=None, then path is assumed to be
                a URL or filesystem path
            img (Optional[string]): usually raw image data. In this case, path will still be stored, but
                a signature will be generated from data in img. If bytestream is False, but img is
                not None, then img is assumed to be the URL or filesystem path. Thus, you can store
                image records with a different 'path' than the actual image location (default None)
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image? If img is None but, this
                argument will be ignored.  If img is not None, and bytestream is False, then the behavior
                is as described in the explanation for the img argument
                (default False)
            metadata (Optional): any other information you want to include, can be nested (default None)
            *args: Variable length argument list to pass to insert_single_record
            **kwargs: Arbitrary keyword arguments to pass to insert_single_record

        """
        rec = make_record(path, self.gis, self.k, self.N, img=img, bytestream=bytestream, metadata=metadata)
        self.insert_single_record(rec, *args, **kwargs)

    def search_image(
        self,
        path: ImageInput,
        all_orientations: bool = False,
        bytestream: bool = False,
        pre_filter: PreFilter = None,
        n_threads: int = 1,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search for matches

        Args:
            path (string): path or image data. If bytestream=False, then path is assumed to be
                a URL or filesystem path. Otherwise, it's assumed to be raw image data
            all_orientations (Optional[boolean]): if True, search for all combinations of mirror
                images, rotations, and color inversions (default False)
            bytestream (Optional[boolean]): will the image be passed as raw bytes?
                That is, is the 'path_or_image' argument an in-memory image?
                (default False)
            pre_filter (Optional[dict]): filters list before applying the matching algorithm
                (default None)
            n_threads (Optional[int]): run orientation searches concurrently in
                this many threads; only relevant when all_orientations=True
                (default 1 = sequential)
            **kwargs: Arbitrary keyword arguments to pass to search_single_record
                (e.g. n_parallel_words for the MongoDB driver)

        Returns:
            a formatted list of dicts representing unique matches, sorted by dist

            For example, if three matches are found::

                [
                 {'dist': 0.069116439263706961,
                  'id': 'AVM37oZq0osmmAxpPvx7',
                  'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg'},
                 {'dist': 0.22484320805049718,
                  'id': 'AVM37nMg0osmmAxpPvx6',
                  'path': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg'},
                 {'dist': 0.42529792112113302,
                  'id': 'AVM37p530osmmAxpPvx9',
                  'path': 'https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg'}
                ]

        """
        records = self._orientation_records(path, all_orientations=all_orientations, bytestream=bytestream)

        if n_threads > 1 and len(records) > 1:
            # each orientation record is an independent dict, so the queries
            # can safely run concurrently (ES/OS clients are thread-safe)
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                per_record = list(pool.map(lambda rec: self.search_single_record(rec, pre_filter=pre_filter, **kwargs), records))
        else:
            per_record = [self.search_single_record(rec, pre_filter=pre_filter, **kwargs) for rec in records]

        return dedupe_results([match for matches in per_record for match in matches])

    def _orientation_records(self, path: ImageInput, all_orientations: bool = False, bytestream: bool = False) -> list[dict[str, Any]]:
        """Build one search record per requested image orientation.

        Pure CPU work (image decode + signature generation); async drivers run
        this in a worker thread.
        """
        img = self.gis.preprocess_image(
            path,
            bytestream=bytestream,
            handle_mpo=self.gis.handle_mpo,
        )
        # default to no transformations
        orientations = [None]

        if all_orientations:
            # initialize an iterator of composed transformations
            inversions = [lambda x: x, lambda x: -x]

            mirrors = [lambda x: x, np.fliplr]

            # an ugly solution for function composition
            rotations = [None, np.rot90, lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3)]

            # cartesian product of all possible orientations
            orientations = product(inversions, rotations, mirrors)

        # try for every possible combination of transformations; if all_orientations=False,
        # this will only take one iteration
        records = []
        for transform in orientations:
            # compose all functions (if not None) and apply on signature
            transformed_img = img
            if transform is not None:
                for f in transform:
                    if f is not None:
                        transformed_img = f(transformed_img)

            # generate the signature; keep the original path on the record
            records.append(make_record(path, self.gis, self.k, self.N, img=transformed_img))

        return records


def dedupe_results(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate ids and sort by dist."""
    ids = set()
    unique = []
    for item in result:
        if item["id"] not in ids:
            unique.append(item)
            ids.add(item["id"])

    return sorted(unique, key=itemgetter("dist"))


def make_record(
    path: ImageInput, gis: ImageSignature, k: int, N: int, img: ImageInput | None = None, bytestream: bool = False, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Makes a record suitable for database insertion.

    Note:
        This non-class version of make_record is provided for
        CPU pooling. Functions passed to worker processes must
        be picklable.

    Args:
        path (string): path or image data. If bytestream=False, then path is assumed to be
            a URL or filesystem path. Otherwise, it's assumed to be raw image data
        gis (ImageSignature): an instance of ImageSignature for generating the
            signature
        k (int): width of words for encoding
        N (int): number of words for encoding
        img (Optional[string]): usually raw image data. In this case, path will still be stored, but
            a signature will be generated from data in img. If bytestream is False, but img is
            not None, then img is assumed to be the URL or filesystem path. Thus, you can store
            image records with a different 'path' than the actual image location (default None)
        bytestream (Optional[boolean]): will the image be passed as raw bytes?
            That is, is the 'path_or_image' argument an in-memory image? If img is None but, this
            argument will be ignored.  If img is not None, and bytestream is False, then the behavior
            is as described in the explanation for the img argument
            (default False)
        metadata (Optional): any other information you want to include, can be nested (default None)

    Returns:
        An image record.

        For example:

        {'path': 'https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg',
         'signature': [0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0 ... ]
         'simple_word_0': 42252475,
         'simple_word_1': 23885671,
         'simple_word_10': 9967839,
         'simple_word_11': 4257902,
         'simple_word_12': 28651959,
         'simple_word_13': 33773597,
         'simple_word_14': 39331441,
         'simple_word_15': 39327300,
         'simple_word_16': 11337345,
         'simple_word_17': 9571961,
         'simple_word_18': 28697868,
         'simple_word_19': 14834907,
         'simple_word_2': 7434746,
         'simple_word_20': 37985525,
         'simple_word_21': 10753207,
         'simple_word_22': 9566120,
         ...
         'metadata': {...}
         }

    """
    record: dict[str, Any] = {"path": path}

    signature = gis.generate_signature(img if img is not None else path, bytestream=bytestream)

    record["signature"] = signature.tolist()

    if metadata:
        record["metadata"] = metadata

    words = get_words(signature, k, N)
    max_contrast(words)

    words = words_to_int(words)

    for i in range(N):
        record["".join(["simple_word_", str(i)])] = words[i].tolist()

    return record


def get_words(array: np.ndarray, k: int, N: int) -> np.ndarray:
    """Gets N words of length k from an array.

    Words may overlap.

    For example, say your image signature is [0, 1, 2, 0, -1, -2, 0, 1] and
    k=3 and N=4. That means we want 4 words of length 3.  For this signature,
    that gives us:

    [0, 1, 2]
    [2, 0, -1]
    [-1, -2, 0]
    [0, 1]

    Args:
        array (numpy.ndarray): array to split into words
        k (int): word length
        N (int): number of words

    Returns:
        an array with N rows of length k

    """
    # generate starting positions of each word
    word_positions = np.linspace(0, array.shape[0], N, endpoint=False).astype("int")

    # check that inputs make sense
    if k > array.shape[0]:
        raise ValueError("Word length cannot be longer than array length")
    if word_positions.shape[0] > array.shape[0]:
        raise ValueError("Number of words cannot be more than array length")

    # create empty words array
    words = np.zeros((N, k)).astype("int8")

    for i, pos in enumerate(word_positions):
        if pos + k <= array.shape[0]:
            words[i] = array[pos : pos + k]
        else:
            tail = array[pos:]
            words[i, : tail.size] = tail  # rest stays zero-padded

    return words


def words_to_int(word_array: np.ndarray) -> np.ndarray:
    """Converts a simplified word to an integer

    Encodes a k-byte word to int (as those returned by max_contrast).
    First digit is least significant.

    Returns dot(word + 1, [1, 3, 9, 27 ...] ) for each word in word_array

    e.g.:
    [ -1, -1, -1] -> 0
    [ 0,   0,  0] -> 13
    [ 0,   1,  0] -> 16

    Args:
        word_array (numpy.ndarray): N x k array

    Returns:
        an array of integers of length N (the integer word encodings)

    """
    width = word_array.shape[1]

    # Three states (-1, 0, 1)
    coding_vector = 3 ** np.arange(width)

    # The 'plus one' here makes all digits positive, so that the
    # integer representation is strictly non-negative and unique
    return np.dot(word_array + 1, coding_vector)


def max_contrast(array: np.ndarray) -> None:
    """Sets all positive values to one and all negative values to -1.

    Needed for first pass lookup on word table.

    Args:
        array (numpy.ndarray): target array

    """
    array[array > 0] = 1
    array[array < 0] = -1

    return None


def normalized_distance(_target_array: np.ndarray, _vec: np.ndarray, nan_value: float = 1.0) -> np.ndarray:
    """Compute normalized distance to many points.

    Computes || vec - b || / ( ||vec|| + ||b||) for every b in target_array

    Args:
        _target_array (numpy.ndarray): N x m array
        _vec (numpy.ndarray): array of size m
        nan_value (Optional[float]): value to replace 0.0/0.0 = nan with
            (default 1.0, to take those featureless images out of contention)

    Returns:
        the normalized distance (float)

    """
    target_array = _target_array.astype(int)
    vec = _vec.astype(int)
    topvec = np.linalg.norm(vec - target_array, axis=1)
    norm1 = np.linalg.norm(vec, axis=0)
    norm2 = np.linalg.norm(target_array, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        finvec = topvec / (norm1 + norm2)
    finvec[np.isnan(finvec)] = nan_value

    return finvec
