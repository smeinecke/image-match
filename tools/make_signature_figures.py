"""Regenerate the step-by-step signature figures used in docs/source/signatures.rst.

Runs the real ImageSignature pipeline on docs/source/_images/MonaLisa_Wikipedia.jpg
and writes one PNG per step into docs/source/_images/ (sig_step_*.png).

Usage:
    uv run python tools/make_signature_figures.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_match.goldberg import ImageSignature
from image_match.signature_database_base import get_words, max_contrast

SRC = Path(__file__).resolve().parent.parent / "docs" / "source" / "_images"
INPUT = SRC / "MonaLisa_Wikipedia.jpg"

# discrete palette for binned values {-2..+2}
BIN_COLORS = {-2: (30, 60, 180), -1: (140, 175, 240), 0: (245, 245, 245), 1: (250, 175, 130), 2: (200, 40, 40)}
# continuous diverging map for raw differences
NEG, ZERO, POS = np.array([30, 60, 180]), np.array([245, 245, 245]), np.array([200, 40, 40])


def diverging(v: float, vmax: float) -> tuple[int, int, int]:
    """Map v in [-vmax, vmax] to blue→white→red."""
    t = max(-1.0, min(1.0, v / vmax))
    rgb = ZERO + t * (POS - ZERO) if t >= 0 else ZERO + (-t) * (NEG - ZERO)
    return tuple(int(c) for c in rgb)


def upscale(img: Image.Image, factor: int) -> Image.Image:
    """Nearest-neighbour upscale so small matrices stay crisp."""
    return img.resize((img.width * factor, img.height * factor), Image.NEAREST)


def matrix_image(mat: np.ndarray, cell: int = 24, grid: bool = True) -> Image.Image:
    """Render a 2D float matrix as a grayscale image with cell borders."""
    norm = np.clip(mat, 0, 1)
    img = upscale(Image.fromarray((norm * 255).astype(np.uint8), "L"), cell).convert("RGB")
    if grid:
        d = ImageDraw.Draw(img)
        for i in range(mat.shape[0] + 1):
            d.line([(0, i * cell), (img.width, i * cell)], fill=(180, 180, 180))
        for j in range(mat.shape[1] + 1):
            d.line([(j * cell, 0), (j * cell, img.height)], fill=(180, 180, 180))
    return img


def compass_image(diff_mat: np.ndarray, micro: int = 15, binned: bool = False) -> Image.Image:
    """Render an nxnx8 differential array as nxn cells of 3x3 mini-blocks.

    Each grid point becomes a 3x3 block; the centre stays neutral and the
    8 surrounding pixels show the difference to that neighbour
    (order: UL, U, UR, L, R, LL, L, LR).
    """
    n = diff_mat.shape[0]
    # scale by the 95th percentile so a few extreme diffs don't wash out the rest
    vmax = float(np.percentile(np.abs(diff_mat), 95)) or 1.0
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    img = Image.new("RGB", (n * 3 * micro, n * 3 * micro), (200, 200, 200))
    d = ImageDraw.Draw(img)
    for i in range(n):
        for j in range(n):
            ox, oy = j * 3 * micro, i * 3 * micro
            for k, (dy, dx) in enumerate(positions):
                v = float(diff_mat[i, j, k])
                color = BIN_COLORS[int(v)] if binned else diverging(v, vmax)
                d.rectangle([ox + dx * micro, oy + dy * micro, ox + (dx + 1) * micro - 1, oy + (dy + 1) * micro - 1], fill=color)
    # thin separators between grid cells
    for i in range(n + 1):
        p = i * 3 * micro
        d.line([(0, p), (img.width, p)], fill=(120, 120, 120))
        d.line([(p, 0), (p, img.height)], fill=(120, 120, 120))
    return img


def words_image(words: np.ndarray, cell_w: int = 14, cell_h: int = 8) -> Image.Image:
    """Render the Nxk words matrix; each row encodes to one integer."""
    n_words, k = words.shape
    img = Image.new("RGB", (k * cell_w, n_words * cell_h), "white")
    d = ImageDraw.Draw(img)
    for i in range(n_words):
        for j in range(k):
            d.rectangle([j * cell_w, i * cell_h, (j + 1) * cell_w - 1, (i + 1) * cell_h - 1], fill=BIN_COLORS[int(words[i, j])])
    return img


def main() -> None:
    """Generate all seven pipeline figures into docs/source/_images/."""
    gis = ImageSignature()
    rgb = Image.open(INPUT).convert("RGB")
    gray = gis.preprocess_image(str(INPUT))

    # step 1 — grayscale
    Image.fromarray((gray * 255).astype(np.uint8), "L").convert("RGB").save(SRC / "sig_step_gray.jpg")

    # step 2 — crop bounds
    limits = gis.crop_image(gray, lower_percentile=gis.lower_percentile, upper_percentile=gis.upper_percentile)
    (top, bottom), (left, right) = limits
    img = rgb.copy()
    ImageDraw.Draw(img).rectangle([left, top, right, bottom], outline=(230, 30, 30), width=4)
    img.save(SRC / "sig_step_crop.jpg")

    # step 3 — grid centres + one PxP window
    x_coords, y_coords = gis.compute_grid_points(gray, n=gis.n, window=limits)
    p = gis.P if gis.P is not None else max(2.0, int(0.5 + min(gray.shape) / 20.0))
    img = rgb.copy()
    d = ImageDraw.Draw(img)
    for x in x_coords:
        for y in y_coords:
            d.ellipse([y - 4, x - 4, y + 4, x + 4], fill=(255, 60, 60), outline=(255, 255, 255))
    cx, cy = int(x_coords[4]), int(y_coords[4])  # centre cell, highlighted
    d.rectangle([cy - p / 2, cx - p / 2, cy + p / 2, cx + p / 2], outline=(40, 200, 60), width=3)
    img.save(SRC / "sig_step_grid.jpg")

    # step 4 — nxn mean grey levels
    avg_grey = gis.compute_mean_level(gray, x_coords, y_coords, P=gis.P)
    matrix_image(avg_grey, cell=36).save(SRC / "sig_step_meanlevel.png")

    # step 5 — raw neighbour differences
    diff_mat = gis.compute_differentials(avg_grey, diagonal_neighbors=gis.diagonal_neighbors)
    compass_image(diff_mat).save(SRC / "sig_step_diff.png")

    # step 6 — thresholded signature (the actual 648-dim vector)
    binned = diff_mat.copy()
    gis.normalize_and_threshold(binned, identical_tolerance=gis.identical_tolerance, n_levels=gis.n_levels)
    compass_image(binned, binned=True).save(SRC / "sig_step_signature.png")

    # step 7 — words before int encoding
    signature = np.ravel(binned).astype("int8")
    words = get_words(signature, 16, 63)  # default k, N from SignatureDatabaseBase
    max_contrast(words)
    words_image(words).save(SRC / "sig_step_words.png")

    # print the numbers quoted in the docs
    print(f"image {gray.shape[1]}x{gray.shape[0]} | crop window rows {top}-{bottom}, cols {left}-{right}")
    print(f"grid x_coords={x_coords.tolist()}\n     y_coords={y_coords.tolist()} | P={p}")
    print(f"signature dims={signature.size}, value histogram={dict(zip(*np.unique(signature, return_counts=True), strict=True))}")
    print(f"words shape={words.shape}")


if __name__ == "__main__":
    main()
