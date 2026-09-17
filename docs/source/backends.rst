Other database backends
=======================
Database backend client libraries are optional extras -- install only the
one you need:

.. code-block:: bash

    $ pip install "image-match[elasticsearch]"        # Elasticsearch driver
    $ pip install "image-match[elasticsearch-async]"  # async Elasticsearch driver
    $ pip install "image-match[opensearch]"           # OpenSearch driver
    $ pip install "image-match[opensearch-async]"     # async OpenSearch driver
    $ pip install "image-match[mongo]"                # MongoDB driver

OpenSearch
----------
``SignatureOpenSearch`` shares the record format and query DSL with the
Elasticsearch driver; only the client library differs (`OpenSearch`_ is
API-compatible with Elasticsearch 7.x for the operations used here):

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
