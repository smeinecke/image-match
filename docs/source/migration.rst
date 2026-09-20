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
    knn engine: ``lucene`` (default), ``faiss``, ``nmslib`` — ``nmslib`` is
    rejected when the target runs OpenSearch 3+ (see below)
``--space-type``
    vector space: ``l2`` (default), ``cosinesimil``, ``innerproduct``, ...
``--data-type``
    knn_vector data type: ``float`` (default) or ``byte`` — signatures are
    int8, so ``byte`` is lossless and ~4x smaller in memory
``--batch-size``
    bulk request size (default 500)
``--slices``
    parallel sliced scrolls over the source index (default 1). A reasonable
    ceiling is the source shard count — but on HDD-backed clusters each
    slice adds random read+write load: 8 slices saturated a two-node HDD
    cluster badly enough to evict the cluster-manager, while 4 ran
    unattended. Watch node io pressure rather than maximizing this.
``--timeout``
    per-request timeout in seconds for both clients (default 30). Raise it
    (e.g. 180) for large shards — scroll fetches and bulk responses on a
    busy cluster can exceed 30 s.
``--skip-existing``
    resume mode: ``_mget``-filter each batch so already-migrated ``_id``\ s
    are not re-indexed. After an interrupted run this turns the re-walk of
    the migrated range into cheap reads instead of tens of millions of
    redundant k-NN upserts (which also trigger HNSW merges).
``--delete-source``
    drop the source index after verification passes

Documents whose signature length doesn't match ``--dimension`` are skipped and
reported, not fatal. If the target index already exists, documents are appended
into it (overwriting by ``_id``). Transient bulk failures (shard
unavailability during merges or node restarts) are retried with exponential
backoff instead of aborting the run. The tool prints detected source/target
server type and version before copying.

.. note::
   k-NN write amplification (Lucene merges + HNSW graph build) is the real
   load, not the scroll. Plan capacity: the hybrid index is roughly 2.5–3×
   the size of the source word index with ``float`` vectors. After the count
   converges, verify recall *before* flipping the application — e.g. search
   a few known images and confirm each returns itself at ``dist 0.0`` (see
   the warning in :doc:`backends`).

Upgrading to OpenSearch 3
-------------------------
The same tool covers an OpenSearch 2 → 3 move — just point ``--target-url``
at the OS3 cluster (word indexes copy verbatim; nothing about the word query
DSL changed):

.. code-block:: bash

    $ uv run python tools/migrate_to_knn.py \
        --source-url http://localhost:9201 --source-index images \
        --target-url http://localhost:9202 --target-index images_knn

OpenSearch 3.0 removed several k-NN index settings (``knn.algo_param.*``) and
**blocks creating new indexes with the ``nmslib`` engine**. Implications:

* ``knn_index_body()`` already emits the OS3-supported style — engine and
  space parameters live in the *field mapping*, not index settings.
* If your OS2 index used ``nmslib``, choose ``--engine faiss`` (or the
  ``lucene`` default) for the OS3 target. The tool fails fast rather than
  producing an index OS3 would reject.
* Existing nmslib indexes keep *working* on a cluster upgraded to OS3 — the
  block applies only to new index creation — but reindex/migration is the
  moment to switch engines.
* ``index.knn: true`` is still required on OS3; the tool sets it.

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
