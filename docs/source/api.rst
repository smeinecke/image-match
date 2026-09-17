API reference
=============

Signature generation
--------------------

.. autoclass:: image_match.goldberg.ImageSignature
    :members:

    .. automethod:: __init__

.. autoexception:: image_match.goldberg.CorruptImageError
    :members:

Driver base
-----------

.. autoclass:: image_match.signature_database_base.SignatureDatabaseBase
    :members:

    .. automethod:: __init__

Module helpers
^^^^^^^^^^^^^^

.. autofunction:: image_match.signature_database_base.make_record
.. autofunction:: image_match.signature_database_base.get_words
.. autofunction:: image_match.signature_database_base.words_to_int
.. autofunction:: image_match.signature_database_base.max_contrast
.. autofunction:: image_match.signature_database_base.normalized_distance
.. autofunction:: image_match.signature_database_base.dedupe_results

Elasticsearch
-------------

Requires the ``elasticsearch`` extra.

.. autoclass:: image_match.elasticsearch_driver.SignatureES
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autofunction:: image_match.elasticsearch_driver.build_word_query
.. autofunction:: image_match.elasticsearch_driver.format_hits

Async
^^^^^

Requires the ``elasticsearch-async`` extra.

.. autoclass:: image_match.elasticsearch_async_driver.AsyncSignatureES
    :members:
    :show-inheritance:

    .. automethod:: __init__

OpenSearch
----------

Requires the ``opensearch`` extra.

.. autoclass:: image_match.opensearch_driver.SignatureOpenSearch
    :members:
    :show-inheritance:

    .. automethod:: __init__

Async
^^^^^

Requires the ``opensearch-async`` extra.

.. autoclass:: image_match.opensearch_async_driver.AsyncSignatureOpenSearch
    :members:
    :show-inheritance:

    .. automethod:: __init__

k-NN drivers
^^^^^^^^^^^^

Search the ``signature`` vector directly with approximate nearest
neighbours instead of word pre-filtering. See :doc:`migration`.

.. autoclass:: image_match.opensearch_knn_driver.SignatureOpenSearchKNN
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: image_match.opensearch_knn_async_driver.AsyncSignatureOpenSearchKNN
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autofunction:: image_match.opensearch_knn_driver.knn_index_body
.. autofunction:: image_match.opensearch_knn_driver.build_knn_query
.. autofunction:: image_match.opensearch_knn_driver.format_knn_hits

MongoDB
-------

Requires the ``mongo`` extra.

.. autoclass:: image_match.mongodb_driver.SignatureMongo
    :members:
    :show-inheritance:

    .. automethod:: __init__

.. autofunction:: image_match.mongodb_driver.get_next_match
