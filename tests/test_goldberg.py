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
    # anchored match= patterns: XX-wrapped or case-mutated messages must fail
    with pytest.raises(AssertionError, match=r"\Acrop_percentiles should be a two-value tuple, or None\Z"):
        ImageSignature(crop_percentiles=(5,))  # type: ignore[arg-type] # not a pair
    with pytest.raises(AssertionError, match=r"\ALower crop_percentiles limit should be > 0 \(-1 given\)\Z"):
        ImageSignature(crop_percentiles=(-1, 95))
    with pytest.raises(AssertionError, match=r"\AUpper crop_percentiles limit should be < 100 \(101 given\)\Z"):
        ImageSignature(crop_percentiles=(5, 101))
    with pytest.raises(AssertionError, match=r"\AUpper crop_percentile limit should be greater than lower limit\.\Z"):
        ImageSignature(crop_percentiles=(95, 5))  # lower >= upper
    with pytest.raises(AssertionError, match=r"\An should be an integer > 1\Z"):
        ImageSignature(n=1.5)  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match=r"\An should be greater than 1 \(1 given\)\Z"):
        ImageSignature(n=1)
    with pytest.raises(AssertionError, match=r"\AP should be an integer >= 1, or None\Z"):
        ImageSignature(P="x")  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match=r"\AP should be greater than 0 \(0 given\)\Z"):
        ImageSignature(P=0)
    with pytest.raises(AssertionError, match=r"\Adiagonal_neighbors should be boolean\Z"):
        ImageSignature(diagonal_neighbors=1)  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match=r"\Afix_ratio should be boolean\Z"):
        ImageSignature(fix_ratio=1)  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match=r"\Aidentical_tolerance should be a number between 1 and 0\Z"):
        ImageSignature(identical_tolerance="x")  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match=r"\Aidentical_tolerance should be greater than zero and less than one \(1\.5 given\)\Z"):
        ImageSignature(identical_tolerance=1.5)
    with pytest.raises(AssertionError, match=r"\An_levels should be an integer\Z"):
        ImageSignature(n_levels=1.5)  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match=r"\An_levels should be > 0 \(0 given\)\Z"):
        ImageSignature(n_levels=0)


def test_init_boundary_values_accepted():
    # exact-boundary inputs must be accepted (kills >=/> and <=/< mutants)
    ImageSignature(crop_percentiles=(0, 95))
    ImageSignature(crop_percentiles=(5, 100))
    ImageSignature(n=2)
    ImageSignature(P=1)  # P >= 1, not P > 1
    ImageSignature(P=5)
    ImageSignature(identical_tolerance=0.0)
    ImageSignature(identical_tolerance=1.0)
    ImageSignature(identical_tolerance=1)  # int accepted
    ImageSignature(n_levels=1)
    ImageSignature(crop_percentiles=None)
    # strictly-inverted range still rejected (kills < -> <= mutant)
    with pytest.raises(AssertionError):
        ImageSignature(crop_percentiles=(50, 50))


def test_init_attributes():
    gis = ImageSignature(n=3, P=5, diagonal_neighbors=False, fix_ratio=True)
    assert gis.P == 5
    assert gis.fix_ratio is True
    assert gis.sig_length == 9 * 4  # n^2 * 4 neighbors
    assert ImageSignature().sig_length == 648
    assert ImageSignature(n=3).sig_length == 9 * 8
    assert ImageSignature().handle_mpo is True


def test_init_crop_percentiles_attribute():
    gis = ImageSignature(crop_percentiles=(10, 80))
    assert gis.crop_percentiles == (10, 80)
    assert gis.lower_percentile == 10
    assert gis.upper_percentile == 80


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
    with pytest.raises(TypeError, match=r"\APath or image required\.\Z"):
        ImageSignature.preprocess_image(12345)  # type: ignore[arg-type]


def test_preprocess_bytestream_rgba_converted():
    """Non-RGB modes must be converted — a convert(None) mutant keeps the
    alpha channel and rgb2gray would blow up on a 4-channel array."""
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8)).save(buf, format="PNG")
    arr = ImageSignature.preprocess_image(buf.getvalue(), bytestream=True)
    assert arr.ndim == 2


def test_preprocess_bytes_path_converts_to_gray(tmp_path):
    """bytes-as-path input: convert('RGB') -> rgb2gray must yield 2D."""
    import numpy as np
    from PIL import Image

    png = tmp_path / "rgba.png"
    Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8)).save(png)
    arr = ImageSignature.preprocess_image(bytes(png))
    assert arr.ndim == 2


def test_preprocess_bytestream_requires_bytes():
    gis = ImageSignature()
    with pytest.raises(TypeError, match=r"\Abytestream=True requires raw image bytes\Z"):
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

    # default fix_ratio=False must NOT force equal ranges
    limits = ImageSignature.crop_image(textured)
    assert limits[0] != limits[1]


def test_crop_image_nonsquare_pins():
    """Pin exact limits on a non-square image — kills shape[0]<->shape[1]
    swaps and percentile-scale mutants."""
    import numpy as np

    img = np.tile(np.arange(60.0)[None, :], (40, 1)) / 60.0
    assert ImageSignature.crop_image(img) == [(2, 38), (3, 57)]
    # fix_ratio picks the wider range for both axes
    assert ImageSignature.crop_image(img, fix_ratio=True) == [(3, 57), (3, 57)]


def test_crop_image_nonsquare_featureless():
    """Featureless non-square image: percentile fallback must use the right
    axis for rows vs columns (kills shape[0]<->shape[1] and 100->101 mutants)."""
    import numpy as np

    limits = ImageSignature.crop_image(np.zeros((40, 60)))
    # rows: int(5/100*40)=2, int(95/100*40)=38; cols: int(5/100*60)=3, int(95/100*60)=57
    assert limits == [(2, 38), (3, 57)]


def test_crop_image_degenerate_equal_limits():
    """A single-pixel feature yields equal lower/upper limits — the '>' guard
    keeps them (a '>=' mutant would reset to the percentile fallback)."""
    import numpy as np

    img = np.zeros((100, 100))
    img[50, 50] = 1.0
    assert ImageSignature.crop_image(img) == [(50, 50), (50, 50)]


def test_crop_image_fix_ratio_equal_ranges():
    """Equal row/col ranges at different positions: '>' (not '>=') picks the
    column range; a row-major image must pick the row range."""
    import numpy as np

    img = np.zeros((100, 100))
    img[10:30, 50:70] = 1.0  # both axes get a ~19px range, at different offsets
    assert ImageSignature.crop_image(img) == [(10, 29), (50, 69)]
    # equal ranges -> else-branch -> column range for both
    assert ImageSignature.crop_image(img, fix_ratio=True) == [(50, 69), (50, 69)]
    # transposed -> row range larger position; still equal -> column wins again
    assert ImageSignature.crop_image(img.T, fix_ratio=True) == [(10, 29), (10, 29)]

    # genuinely larger row range -> if-branch -> row range for both
    img2 = np.zeros((100, 100))
    img2[5:80, 40:60] = 1.0
    row, col = ImageSignature.crop_image(img2)
    assert (row[1] - row[0]) > (col[1] - col[0])
    assert ImageSignature.crop_image(img2, fix_ratio=True) == [row, row]


def test_compute_grid_points_default_window():
    import numpy as np

    x, y = ImageSignature.compute_grid_points(np.zeros((100, 100)), n=9)
    assert len(x) == len(y) == 9
    # default n is 9
    x, y = ImageSignature.compute_grid_points(np.zeros((100, 100)))
    assert len(x) == len(y) == 9


def test_compute_grid_points_pins():
    """Exact grid coordinates on a non-square image — kills window/shape-swap
    and linspace dtype mutants."""
    import numpy as np

    # 41x61 so a window-start mutant (0->1) shifts the int-truncated coords
    x, y = ImageSignature.compute_grid_points(np.zeros((41, 61)), n=4)
    # linspace(0, 41, 6, int)[1:-1] / linspace(0, 61, 6, int)[1:-1]
    assert x.tolist() == [8, 16, 24, 32]
    assert y.tolist() == [12, 24, 36, 48]
    assert x.dtype.kind == "i" and y.dtype.kind == "i"
    # explicit window is honored
    x, y = ImageSignature.compute_grid_points(np.zeros((40, 60)), n=2, window=[(10, 30), (20, 50)])
    assert x.tolist() == [16, 23]
    assert y.tolist() == [30, 40]


def test_compute_differentials_default_diagonal():
    import numpy as np

    m = np.random.default_rng(0).random((4, 4))
    # default diagonal_neighbors=True -> 8 neighbors per grid point
    assert ImageSignature.compute_differentials(m).shape == (4, 4, 8)
    assert ImageSignature.compute_differentials(m, diagonal_neighbors=False).shape == (4, 4, 4)


def test_normalize_and_threshold_default_levels():
    import numpy as np

    # n_levels=2 bins differences into {-2..2}; n_levels=3 would reach +-3
    arr = np.array([[[0.0, 0.1, 0.5, 1.0, -0.1, -0.5, -1.0, 2.0]]])
    ImageSignature.normalize_and_threshold(arr)
    assert set(np.unique(arr)) <= {-2.0, -1.0, 0.0, 1.0, 2.0}
    assert arr.max() == 2.0 and arr.min() == -2.0


def test_normalize_and_threshold_featureless():
    import numpy as np

    # all values below identical_tolerance -> early exit, stays zeroed
    arr = np.zeros((5, 5, 8))
    ImageSignature.normalize_and_threshold(arr)
    assert np.all(arr == 0.0)


def test_generate_signature_forwards_parameters(monkeypatch):
    import numpy as np

    calls = {}
    orig_crop = ImageSignature.crop_image
    orig_grid = ImageSignature.compute_grid_points
    orig_mean = ImageSignature.compute_mean_level
    orig_diff = ImageSignature.compute_differentials
    orig_norm = ImageSignature.normalize_and_threshold

    def crop(image, **kwargs):
        calls["crop"] = kwargs
        return orig_crop(image, **kwargs)

    def grid(image, **kwargs):
        calls["grid"] = kwargs
        return orig_grid(image, **kwargs)

    def mean(image, x, y, **kwargs):
        calls["mean"] = kwargs
        return orig_mean(image, x, y, **kwargs)

    def diff(matrix, **kwargs):
        calls["diff"] = kwargs
        return orig_diff(matrix, **kwargs)

    def norm(difference_array, **kwargs):
        calls["norm"] = kwargs
        return orig_norm(difference_array, **kwargs)

    monkeypatch.setattr(ImageSignature, "crop_image", staticmethod(crop))
    monkeypatch.setattr(ImageSignature, "compute_grid_points", staticmethod(grid))
    monkeypatch.setattr(ImageSignature, "compute_mean_level", staticmethod(mean))
    monkeypatch.setattr(ImageSignature, "compute_differentials", staticmethod(diff))
    monkeypatch.setattr(ImageSignature, "normalize_and_threshold", staticmethod(norm))

    gis = ImageSignature(
        n=3,
        P=7,
        crop_percentiles=(3, 90),
        diagonal_neighbors=False,
        identical_tolerance=0.01,
        n_levels=1,
        fix_ratio=True,
    )
    signature = gis.generate_signature(np.random.default_rng(0).random((100, 100)))
    assert isinstance(signature, np.ndarray)
    assert signature.dtype == np.int8
    assert len(signature) == 3 * 3 * 4

    assert calls["crop"] == {"lower_percentile": 3, "upper_percentile": 90, "fix_ratio": True}
    assert calls["grid"]["n"] == 3
    assert calls["mean"] == {"P": 7}
    assert calls["diff"] == {"diagonal_neighbors": False}
    assert calls["norm"] == {"identical_tolerance": 0.01, "n_levels": 1}


def test_preprocess_bytes_imread_fallback(monkeypatch):
    from unittest.mock import Mock

    import numpy as np

    import image_match.goldberg as goldberg_mod

    monkeypatch.setattr(goldberg_mod.Image, "open", Mock(side_effect=OSError("corrupt")))
    fake_imread = Mock(return_value=np.zeros((4, 4)))
    monkeypatch.setattr(goldberg_mod, "imread", fake_imread)

    out = ImageSignature.preprocess_image(b"not-an-image")
    fake_imread.assert_called_once_with(b"not-an-image", as_gray=True)
    assert out.shape == (4, 4)


def test_compute_mean_level_edge_clamp():
    """The window clamp is max(x - p/2, 0) — a mutant clamping at 1 shifts the
    window off the border and changes the mean."""
    import numpy as np

    img = np.zeros((4, 4))
    img[0, :] = 8.0
    img[:, 0] = 2.0
    # x=y=1, P=2 -> window rows [0:2], cols [0:2] = [[2,8],[2,0]] -> mean 3.0
    out = ImageSignature.compute_mean_level(img, np.array([1]), np.array([1]), P=2)
    assert out.tolist() == [[3.0]]


def test_compute_mean_level_window_size():
    import numpy as np

    # default: p = max(2.0, int(0.5 + min(shape) / 20.0))
    img40 = np.arange(1600, dtype=float).reshape(40, 40) ** 2
    out = ImageSignature.compute_mean_level(img40, np.array([20]), np.array([20]))
    assert out[0, 0] == np.mean(img40[19:21, 19:21])  # p=2

    img50 = np.arange(2500, dtype=float).reshape(50, 50) ** 2
    out50 = ImageSignature.compute_mean_level(img50, np.array([25]), np.array([25]))
    assert out50[0, 0] == np.mean(img50[23:26, 23:26])  # p=3

    # explicit P is honored over the auto formula
    out4 = ImageSignature.compute_mean_level(img50, np.array([25]), np.array([25]), P=4)
    assert out4[0, 0] == np.mean(img50[23:27, 23:27])
    assert out4[0, 0] != out50[0, 0]

    # tall image: the x-window bound must clamp to rows (shape[0]), not columns
    tall = np.arange(2400, dtype=float).reshape(60, 40) ** 2
    out_tall = ImageSignature.compute_mean_level(tall, np.array([59]), np.array([20]))
    assert not np.isnan(out_tall).any()
    assert out_tall[0, 0] == np.mean(tall[58:60, 19:21])


def test_compute_differentials_values():
    import numpy as np

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    d = ImageSignature.compute_differentials(m)
    # slot order: upper-left, upper, upper-right, left, right, lower-left, lower, lower-right
    assert d[1, 1].tolist() == [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0]
    assert d[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0, -0.0, -3.0, -4.0]


def test_normalize_and_threshold_one_sided():
    import numpy as np

    # all-negative differences: positive branch must be skipped entirely
    arr = np.array([[[-1.0, -0.5, -0.2, -0.8]]])
    ImageSignature.normalize_and_threshold(arr, identical_tolerance=0.01, n_levels=2)
    assert np.all(arr <= 0.0)
    assert set(np.unique(arr)) <= {-2.0, -1.0, 0.0}

    arr = np.array([[[1.0, 0.5, 0.2, 0.8]]])
    ImageSignature.normalize_and_threshold(arr, identical_tolerance=0.01, n_levels=2)
    assert np.all(arr >= 0.0)
    assert set(np.unique(arr)) <= {0.0, 1.0, 2.0}


def test_normalize_and_threshold_tolerance_boundary():
    """A diff exactly equal to identical_tolerance must NOT be masked
    ('<' not '<=')."""
    import numpy as np

    tol = 0.1
    arr = np.array([[[tol, 2 * tol, 3 * tol]]])
    ImageSignature.normalize_and_threshold(arr, identical_tolerance=tol, n_levels=2)
    # tol itself survives the mask and gets binned to a nonzero level
    assert arr[0, 0, 0] != 0.0


def test_static_normalized_distance():
    import numpy as np

    a = np.array([1.0, 2.0])
    assert ImageSignature.normalized_distance(a, a) == 0.0
    # two zero vectors -> 0/0 -> nan (the base-module variant replaces with nan_value)
    assert np.isnan(ImageSignature.normalized_distance(np.zeros(2), np.zeros(2)))
    # fractional inputs are truncated to int — asymmetric pins kill astype(None)
    assert ImageSignature.normalized_distance(np.array([0.9, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert ImageSignature.normalized_distance(np.array([1.0, 0.0]), np.array([0.9, 0.0])) == 1.0


def test_static_normalized_distance_no_warnings():
    """np.errstate must suppress the 0/0 invalid warning — dropping 'invalid'
    makes it escape."""
    import warnings

    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(ImageSignature.normalized_distance(np.zeros(3), np.zeros(3)))
