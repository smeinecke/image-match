import pytest
from numpy import array_equal, ndarray

from image_match.goldberg import CorruptImageError, ImageSignature

from .helpers import TEST_IMG_URL as test_img_url


def test_load_from_url():
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


def test_all_inputs_same_sig():
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
