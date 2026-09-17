Quickstart
==========

The fastest path from zero to a working image search. See :doc:`start` for
full installation options and :doc:`searches` for the complete API surface.

Install
-------

.. code-block:: bash

    $ pip install "image-match[opensearch]"   # or [elasticsearch], [mongo]

Start a backend
---------------

The repository ships a Docker Compose stack. OpenSearch 2.x listens on
``:9201``, Elasticsearch on ``:9200``, MongoDB on ``:27017``:

.. code-block:: bash

    $ docker compose up -d opensearch

Index and search
----------------

.. code-block:: python

    from opensearchpy import OpenSearch
    from image_match.opensearch_driver import SignatureOpenSearch

    ses = SignatureOpenSearch(OpenSearch("http://localhost:9201"))

    ses.add_image("photo_a.jpg")
    ses.add_image("photo_b.jpg")
    ses.add_image("https://example.org/photo_c.jpg")

    hits = ses.search_image("photo_a.jpg")

Each hit is a dict — at minimum ``id``, ``path`` and ``dist``. ``dist`` is the
normalized distance between signatures: ``0.0`` is identical, ``< 0.40`` is
very likely the same image, and hits above ``distance_cutoff`` (default
``0.45``) are filtered out entirely. Add ``refresh_after=True`` to
``add_image`` when you need a record searchable immediately.

Approximate nearest neighbour search (OpenSearch)
-------------------------------------------------

For very large indexes, the k-NN drivers search the signature vector
directly instead of pre-filtering on words:

.. code-block:: python

    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN

    knn = SignatureOpenSearchKNN(OpenSearch("http://localhost:9201"), index="images_knn")
    knn.index_collection()          # create the knn_vector index
    knn.add_image("photo_a.jpg")
    hits = knn.search_image("photo_a.jpg")

Existing indexes can be migrated in place — see :doc:`migration`.

No backend? Signatures only
---------------------------

Signature generation and comparison need no database at all:

.. code-block:: python

    from image_match.goldberg import ImageSignature

    gis = ImageSignature()
    a = gis.generate_signature("photo_a.jpg")
    b = gis.generate_signature("photo_b.jpg")
    print(gis.normalized_distance(a, b))   # e.g. 0.22 → likely a match
