import pytest
from numpy import array_equal, ndarray

from image_match.goldberg import CorruptImageError, ImageSignature

from .helpers import TEST_IMG_URL as test_img_url


def test_load_from_url(requires_download):
    gis = ImageSignature()
    sig = gis.generate_signature(test_img_url)
    assert type(sig) is ndarray
    assert sig.shape == (648,)


def test_load_from_file():
    gis = ImageSignature()
    sig = gis.generate_signature("test.jpg")
    assert type(sig) is ndarray
    assert sig.shape == (648,)


def test_load_from_unicode_path():
    path = "test.jpg"
    gis = ImageSignature()
    sig = gis.generate_signature(path)
    assert type(sig) is ndarray
    assert sig.shape == (648,)


def test_load_from_stream():
    gis = ImageSignature()
    with open("test.jpg", "rb") as f:
        sig = gis.generate_signature(f.read(), bytestream=True)
        assert type(sig) is ndarray
        assert sig.shape == (648,)


def test_load_from_corrupt_stream():
    gis = ImageSignature()
    with pytest.raises(CorruptImageError):
        gis.generate_signature(b"corrupt", bytestream=True)


def test_all_inputs_same_sig(requires_download):
    gis = ImageSignature()
    sig1 = gis.generate_signature(test_img_url)
    sig2 = gis.generate_signature("test_url.jpg")
    with open("test_url.jpg", "rb") as f:
        sig3 = gis.generate_signature(f.read(), bytestream=True)

    assert array_equal(sig1, sig2)
    assert array_equal(sig2, sig3)


def test_identity():
    gis = ImageSignature()
    sig = gis.generate_signature("test.jpg")
    dist = gis.normalized_distance(sig, sig)
    assert dist == 0.0


def test_difference():
    gis = ImageSignature()
    sig1 = gis.generate_signature("test.jpg")
    sig2 = gis.generate_signature("test_diff.jpg")
    dist = gis.normalized_distance(sig1, sig2)
    assert dist == 0.42672771706789686


# --- constructor validation -------------------------------------------------


def test_init_validation():
    with pytest.raises(AssertionError):
        ImageSignature(crop_percentiles=(5,))  # type: ignore[arg-type] # not a pair
    with pytest.raises(AssertionError):
        ImageSignature(crop_percentiles=(-1, 95))
    with pytest.raises(AssertionError):
        ImageSignature(crop_percentiles=(95, 5))  # lower >= upper
    with pytest.raises(AssertionError):
        ImageSignature(n=1.5)  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        ImageSignature(n=1)
    with pytest.raises(AssertionError):
        ImageSignature(P="x")  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        ImageSignature(P=0)
    with pytest.raises(AssertionError):
        ImageSignature(diagonal_neighbors=1)  # type: ignore[arg-type]


def test_init_crop_percentiles_none():
    gis = ImageSignature(crop_percentiles=None)
    assert gis.lower_percentile == 0
    assert gis.upper_percentile == 100
    # no-crop path through generate_signature -> compute_grid_points(window=None)
    sig = gis.generate_signature("test.jpg")
    assert sig.shape == (648,)


# --- preprocess_image input types -------------------------------------------


def test_preprocess_pathlike():
    from pathlib import Path

    arr = ImageSignature.preprocess_image(Path("test.jpg"))
    assert arr.ndim == 2


def test_preprocess_bytes_path():
    # bytes input without bytestream -> treated as a bytes-encoded path
    arr = ImageSignature.preprocess_image(b"test.jpg")
    assert arr.ndim == 2


def test_preprocess_ndarray_color_and_gray():
    import numpy as np
    from PIL import Image

    color = np.array(Image.open("test.jpg").convert("RGB"))
    gray = ImageSignature.preprocess_image(color)
    assert gray.ndim == 2

    # already-grayscale ndarray is returned as-is
    back = ImageSignature.preprocess_image(gray)
    assert back.shape == gray.shape


def test_preprocess_invalid_type():
    with pytest.raises(TypeError):
        ImageSignature.preprocess_image(12345)  # type: ignore[arg-type]


def test_preprocess_bytestream_requires_bytes():
    gis = ImageSignature()
    with pytest.raises(TypeError, match="bytestream"):
        gis.generate_signature("test.jpg", bytestream=True)


def test_preprocess_truncated_stream():
    gis = ImageSignature()
    with open("test.jpg", "rb") as f:
        data = f.read()[:100]  # truncated jpeg
    with pytest.raises(CorruptImageError):
        gis.generate_signature(data, bytestream=True)


def test_preprocess_svg_stream():
    pytest.importorskip("cairosvg", reason="cairosvg not installed (install the 'extra' extra)")
    gis = ImageSignature()
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><rect width="20" height="20" fill="red"/></svg>'
    sig = gis.generate_signature(svg, bytestream=True)
    assert sig.shape == (648,)


# --- crop / grid / threshold helpers ----------------------------------------


def test_crop_image_featureless_and_fix_ratio():
    import numpy as np

    # uniform image -> featureless -> default percentile region
    flat = np.zeros((50, 50))
    limits = ImageSignature.crop_image(flat)
    assert limits == [(2, 47), (2, 47)]

    # fix_ratio picks the larger range for both axes
    textured = np.tile(np.arange(50.0)[:, None], (1, 80))  # varies along x only
    limits = ImageSignature.crop_image(textured, fix_ratio=True)
    assert limits[0] == limits[1]


def test_compute_grid_points_default_window():
    import numpy as np

    x, y = ImageSignature.compute_grid_points(np.zeros((100, 100)), n=9)
    assert len(x) == len(y) == 9


def test_normalize_and_threshold_featureless():
    import numpy as np

    # all values below identical_tolerance -> early exit, stays zeroed
    arr = np.zeros((5, 5, 8))
    ImageSignature.normalize_and_threshold(arr)
    assert np.all(arr == 0.0)


def test_static_normalized_distance():
    import numpy as np

    a = np.array([1.0, 2.0])
    assert ImageSignature.normalized_distance(a, a) == 0.0
    # two zero vectors -> 0/0 -> nan (the base-module variant replaces with nan_value)
    assert np.isnan(ImageSignature.normalized_distance(np.zeros(2), np.zeros(2)))
