Other database backends
=======================
Database backend client libraries are optional extras -- install only the
one you need:

.. code-block:: bash

    $ pip install "image-match[elasticsearch]"        # Elasticsearch driver
    $ pip install "image-match[elasticsearch-async]"  # async Elasticsearch driver
    $ pip install "image-match[opensearch]"           # OpenSearch driver (+ k-NN)
    $ pip install "image-match[opensearch-async]"     # async OpenSearch driver (+ k-NN)
    $ pip install "image-match[mongo]"                # MongoDB driver

OpenSearch
----------
``SignatureOpenSearch`` shares the record format and query DSL with the
Elasticsearch driver; only the client library differs (`OpenSearch`_ is
API-compatible with Elasticsearch 7.x for the operations used here). Both
OpenSearch 2.x and 3.x servers are supported — the ``opensearch`` extra pulls
``opensearch-py>=2.4``, whose 3.x client line covers server versions 1.x–3.x:

.. code-block:: python

    from image_match.opensearch_driver import SignatureOpenSearch
    from opensearchpy import OpenSearch

    ses = SignatureOpenSearch(OpenSearch("http://localhost:9200"))

Async drivers
-------------
``AsyncSignatureES`` and ``AsyncSignatureOpenSearch`` mirror the synchronous
drivers, but every database-touching method is a coroutine. Image decoding and
signature generation run in a worker thread via ``asyncio.to_thread``, so the
event loop is never blocked. Use these when embedding image-match in async
services such as FastAPI:

.. code-block:: python

    from image_match.elasticsearch_async_driver import AsyncSignatureES
    from elasticsearch import AsyncElasticsearch

    ses = AsyncSignatureES(AsyncElasticsearch("http://localhost:9200"))

    # in an async endpoint / handler:
    await ses.add_image("cat.jpg", refresh_after=True)
    matches = await ses.search_image("cat.jpg")

    # don't forget to close the client on shutdown:
    await ses.es.close()

The async drivers expose the same API — ``add_image``, ``search_image``,
``search_single_record``, ``insert_single_record`` and ``delete_duplicates`` —
plus the same constructor options (``index``, ``timeout``, ``size``,
``delete_duplicates_limit``). For OpenSearch, swap in
``AsyncSignatureOpenSearch`` with an ``AsyncOpenSearch`` client. MongoDB has no
async driver; wrap the sync driver in ``asyncio.to_thread`` instead if needed.

OpenSearch k-NN
---------------
``SignatureOpenSearchKNN`` (and ``AsyncSignatureOpenSearchKNN``) use
OpenSearch's native approximate nearest-neighbour search on the ``signature``
vector directly, instead of the word-overlap disjunction. The record format is
identical — records already store ``signature``, so no data transformation is
needed, only a different index mapping:

.. code-block:: python

    from opensearchpy import OpenSearch
    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN, knn_index_body

    client = OpenSearch("http://localhost:9201")
    client.indices.create(index="images_knn", body=knn_index_body(648))

    ses = SignatureOpenSearchKNN(client, index="images_knn")
    ses.add_image("cat.jpg", refresh_after=True)
    matches = ses.search_image("cat.jpg")

The index must be created with ``index.knn: true`` and the ``signature`` field
mapped as ``knn_vector`` — ``knn_index_body()`` produces exactly that (648 is
the signature length for the default ``n_grid=9``). All other fields stay
dynamically mapped, so ``simple_word_*`` fields still index normally and the
classic word drivers keep working on the same index — migrations are
reversible (see :doc:`migration`).

The knn candidates are rescored client-side with the usual normalized
distance, so ``dist``/``distance_cutoff`` semantics are unchanged. The
difference is recall: HNSW is *approximate* — ``size`` controls the candidate
count ``k``; raise it for better recall on large indexes.

Engine and data-type choices via ``knn_index_body(..., engine=..., space_type=..., data_type=...)``:

* ``engine``: ``lucene`` (default, no native dependency) or ``faiss``.
  ``nmslib`` is deprecated and **rejected for new indexes on OpenSearch 3**
  — only use it for OpenSearch 2 targets.
* ``data_type``: ``float`` (default) or ``byte``. Signatures are int8, so
  ``byte`` is lossless and ~4x smaller in memory (requires OpenSearch ≥ 2.9).

Both OpenSearch 2.x and 3.x accept the mapping ``knn_index_body()`` produces
— method parameters live in the field mapping, the style OpenSearch 3
requires (index-level ``knn.algo_param.*`` settings were removed in 3.0).

MongoDB
-------
For demonstration purposes we include also a `MongoDB`_ driver:

.. code-block:: python

    from image_match.mongodb_driver import SignatureMongo
    from pymongo import MongoClient

    client = MongoClient(connect=False)
    c = client.images.images

    ses = SignatureMongo(c)

now you can use the same functionality as above like ``ses.add_image(...)``.

We tried to separate signature logic from the database insertion/search as much
as possible.  To write your own database backend, you can inherit from the
``SignatureDatabaseBase`` class and override the appropriate methods:

.. code-block:: python

    from image_match.signature_database_base import SignatureDatabaseBase
    # other relevant imports

    class MySignatureBackend(SignatureDatabaseBase):
    
        # if you need to do some setup, override __init__
        def __init__(self, myarg1, myarg2, *args, **kwargs):
            # do some initializing stuff here if necessary
            # ...
            super().__init__(*args, **kwargs)
    
        # you MUST implement these two functions
        def search_single_record(self, rec):
            # should query your database given a record generated from
            # signature_database_base.make_record
            # ...
            # should return a list of dicts like 
            # [{'id': 'some_unique_id_from_db',
            #   'dist': 0.109234,
            #   'path': 'url/or/filepath'},
            #  {...}, ...]
            # you can have other keys, but you need at least id and dist
            return formatted_results
    
        def insert_single_record(self, rec):
            # if your database driver or instance can accept a dict as input,
            # this should be very simple
    
        # ...

Unfortunately, implementing a good ``search_single_record`` function does
require some knowledge of `the search algorithm`_. You can also look at the two
included database drivers for guidelines.



.. _MongoDB: https://www.mongodb.org/
.. _OpenSearch: https://opensearch.org/
.. _the search algorithm: http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps
