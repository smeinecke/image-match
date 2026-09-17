Image signatures and distances
==============================
Consider these two photographs of the `Mona Lisa`_:

.. image:: _images/MonaLisa_Wikipedia.jpg

(credit:
`Wikipedia <https://en.wikipedia.org/wiki/Mona_Lisa#/media/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg>`__
Public domain)

.. image:: _images/MonaLisa_WikiImages.jpg

(credit:
`WikiImages <https://pixabay.com/en/mona-lisa-painting-art-oil-painting-67506/>`_
Public domain)

Though it's obvious to any human observer that this is the same image, we can
find a number of subtle differences: the dimensions, palette, lighting and so
on are different in each image. ``image_match`` will give us numerical
comparison:

.. code-block:: python

    from image_match.goldberg import ImageSignature
    gis = ImageSignature()
    a = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg/687px-Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
    b = gis.generate_signature('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
    gis.normalized_distance(a, b)

Returns ``0.22095170140933634``. Normalized distances of less than ``0.40`` are
very likely matches. If we try this again against a dissimilar image, say,
Caravaggio's `Supper at Emmaus <https://en.wikipedia.org/wiki/Supper_at_Emmaus_(Caravaggio),_London>`_:

.. image:: _images/Caravaggio_Wikipedia.jpg

(credit: `Wikipedia <https://en.wikipedia.org/wiki/Caravaggio#/media/File:Caravaggio_-_Cena_in_Emmaus.jpg>`__ Public domain)

against one of the Mona Lisa photographs:

.. code-block:: python

    c = gis.generate_signature('https://upload.wikimedia.org/wikipedia/commons/e/e0/Caravaggio_-_Cena_in_Emmaus.jpg')
    gis.normalized_distance(a, c)

Returns ``0.68446275381507249``, almost certainly not a match. ``image_match``
doesn't have to generate a signature from a URL; a file-path or even an
in-memory bytestream will do (be sure to specify ``bytestream=True`` in the
latter case).

Now consider this subtly-modified version of the Mona Lisa:

.. image:: _images/MonaLisa_Remix_Flickr.jpg

(credit: `Michael Russell <https://www.flickr.com/photos/planetrussell/6814444991>`_ `Attribution-ShareAlike 2.0 Generic <https://creativecommons.org/licenses/by-sa/2.0/>`_)

How similar is it to our original Mona Lisa?

.. code-block:: python

    d = gis.generate_signature('https://c2.staticflickr.com/8/7158/6814444991_08d82de57e_z.jpg')
    gis.normalized_distance(a, d)

This gives us ``0.42557196987336648``. So markedly different than the two
original Mona Lisas, but considerably closer than the Caravaggio.


How a signature is computed
---------------------------

``generate_signature`` follows the Goldberg et al. paper: reduce the image to
an n×n grid of local grey levels, take differences between neighbours, and
quantize them into a compact integer vector.

.. mermaid::

    flowchart TD
        subgraph sig["Signature generation — image_match.goldberg"]
            direction TB
            A["input image<br/>path · URL · array · bytes"] --> B["preprocess_image<br/>decode → greyscale float array"]
            B --> C["crop_image<br/>percentile bounds (5–95%)<br/>drop featureless borders"]
            C --> D["compute_grid_points<br/>n×n grid centres (9×9)"]
            D --> E["compute_mean_level<br/>mean grey per P×P window<br/>→ n×n matrix"]
            E --> F["compute_differentials<br/>diffs vs. 8 neighbours<br/>→ n×n×8"]
            F --> G["normalize_and_threshold<br/>bin to {-2 … +2}"]
            G --> H["signature<br/>int8 vector, 9·9·8 = 648 dims"]
        end

        subgraph idx["Indexing — make_record"]
            direction TB
            H --> I["get_words<br/>N = 63 overlapping words × k = 16"]
            I --> J["max_contrast<br/>collapse to {-1, 0, 1}"]
            J --> K["words_to_int<br/>each word → one integer"]
            K --> L["record:<br/>path · signature · metadata<br/>simple_word_0 … 62"]
            L --> M[("backend index")]
        end

        subgraph qry["Search — search_image"]
            direction TB
            Q["query image → same pipeline → record"] --> R["word query:<br/>match any simple_word_*"]
            M -.-> R
            R --> S["candidates: normalized_distance<br/>on full 648-dim signatures"]
            S --> T["keep dist &lt; distance_cutoff (0.45)<br/>dedupe → sort by dist"]
        end

The ``simple_word_*`` integers act as a cheap pre-filter: matching any one
word already makes a record a candidate, so the expensive full-signature
distance is only computed on a small candidate set. This is what lets the
index scale to billions of images. With ``all_orientations=True`` the query
is expanded to up to 8 rotations/mirrors/inversions of the same image.

The OpenSearch k-NN drivers skip the word pre-filter entirely — the 648-dim
``signature`` is stored as a ``knn_vector`` and searched directly with
approximate nearest neighbours. See :doc:`migration` for moving an existing
word index to a k-NN index.


.. _Mona Lisa: https://en.wikipedia.org/wiki/Mona_Lisa
