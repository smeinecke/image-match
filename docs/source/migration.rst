Migrating to the OpenSearch k-NN index
======================================
The k-NN drivers (``SignatureOpenSearchKNN`` / ``AsyncSignatureOpenSearchKNN``)
search the ``signature`` vector directly with approximate nearest neighbours.
Records already store ``signature``, so existing data needs no transformation —
only the *index mapping* must change (``signature`` becomes a ``knn_vector``
field and the index needs ``index.knn: true``).

Because a field's mapping cannot be changed on an existing index, migration
creates a **new hybrid index**: ``signature`` is mapped as ``knn_vector`` and
every other field keeps dynamic mapping. That means:

* ``simple_word_*`` fields index as numbers just like before — the classic word
  drivers keep working on the migrated index, so you can roll back or run both
  drivers side by side.
* Document ``_id`` values are preserved — re-running the migration is
  idempotent.
* The source index is never modified (unless you pass ``--delete-source``
  after a successful verification).

Automated migration
-------------------
``tools/migrate_to_knn.py`` handles the copy: it scans the source index with a
scroll, validates that each document's signature matches the expected
dimension, bulk-writes into the new k-NN index, then verifies the result.

Elasticsearch → OpenSearch:

.. code-block:: bash

    $ uv run python tools/migrate_to_knn.py \
        --source-url http://localhost:9200 --source-index images \
        --target-url http://localhost:9201 --target-index images_knn

OpenSearch word index → OpenSearch k-NN index (same or different cluster):

.. code-block:: bash

    $ uv run python tools/migrate_to_knn.py \
        --source-url http://localhost:9201 --source-index images \
        --target-url http://localhost:9201 --target-index images_knn

Options:

``--dimension``
    signature length, default ``648`` (default ``n_grid=9``)
``--engine``
    knn engine: ``lucene`` (default), ``faiss``, ``nmslib``
``--space-type``
    vector space: ``l2`` (default), ``cosinesimil``, ``innerproduct``, ...
``--batch-size``
    bulk request size (default 500)
``--delete-source``
    drop the source index after verification passes

Documents whose signature length doesn't match ``--dimension`` are skipped and
reported, not fatal. If the target index already exists, documents are appended
into it (overwriting by ``_id``).

Manual alternative
------------------
You can do the same by hand: create the target index with
``knn_index_body(dimension)`` (see :doc:`backends`), then reindex. The
``_reindex`` API alone cannot convert a field's type, so a copy via
scroll + bulk (what the tool does) is required — any client-side copy works.
Alternatively, re-add all images with the new driver: ``add_image`` produces
the same record regardless of backend.

Switching over
--------------
Point your code at the new index and driver:

.. code-block:: python

    from image_match.opensearch_knn_driver import SignatureOpenSearchKNN
    ses = SignatureOpenSearchKNN(client, index="images_knn")

If something goes wrong, switch back to the word driver on the *same* index —
the word fields are still populated and indexed.
