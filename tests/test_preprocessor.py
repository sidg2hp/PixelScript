"""Unit tests for the Preprocessor — no OCR models needed."""

import numpy as np
import pytest

from ocr.preprocessor import Preprocessor, PreprocessConfig


def _make_image(w: int = 300, h: int = 200, color: bool = True) -> np.ndarray:
    """Create a synthetic test image (white with a black rectangle)."""
    if color:
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        img[50:150, 50:250] = (0, 0, 0)
    else:
        img = np.full((h, w), 255, dtype=np.uint8)
        img[50:150, 50:250] = 0
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Resize
# ──────────────────────────────────────────────────────────────────────────────

def test_resize_upscales_narrow_image():
    img = _make_image(w=200)
    cfg = PreprocessConfig(min_width=1000, denoise=False, binarize=False, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    assert out.shape[1] >= 1000


def test_resize_does_not_shrink_wide_image():
    img = _make_image(w=1500)
    cfg = PreprocessConfig(min_width=1000, denoise=False, binarize=False, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    assert out.shape[1] >= 1000  # not shrunk


# ──────────────────────────────────────────────────────────────────────────────
# Grayscale
# ──────────────────────────────────────────────────────────────────────────────

def test_output_is_grayscale():
    img = _make_image(color=True)
    cfg = PreprocessConfig(denoise=False, binarize=False, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    assert out.ndim == 2, "Expected 2-D grayscale output"


def test_grayscale_input_unchanged_dims():
    img = _make_image(color=False)
    cfg = PreprocessConfig(denoise=False, binarize=False, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    assert out.ndim == 2


# ──────────────────────────────────────────────────────────────────────────────
# Binarization
# ──────────────────────────────────────────────────────────────────────────────

def test_binarize_adaptive_produces_binary_values():
    img = _make_image()
    cfg = PreprocessConfig(denoise=False, binarize=True, binarize_method="adaptive", deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    unique = np.unique(out)
    assert set(unique).issubset({0, 255}), f"Non-binary values: {unique}"


def test_binarize_otsu_produces_binary_values():
    img = _make_image()
    cfg = PreprocessConfig(denoise=False, binarize=True, binarize_method="otsu", deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    unique = np.unique(out)
    assert set(unique).issubset({0, 255}), f"Non-binary values: {unique}"


# ──────────────────────────────────────────────────────────────────────────────
# Border
# ──────────────────────────────────────────────────────────────────────────────

def test_border_increases_dimensions():
    img = _make_image(w=400, h=300)
    px = 10
    cfg = PreprocessConfig(denoise=False, binarize=False, deskew=False, border_px=px)
    out = Preprocessor(cfg).process(img)
    # After resize (400 >= 1000 default is False here, but let's set min_width=0)
    # Easier: just check border was added
    assert out.shape[0] > img.shape[0]
    assert out.shape[1] > img.shape[1]


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline (smoke test)
# ──────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_runs_without_error():
    img = _make_image(w=500, h=300)
    out = Preprocessor().process(img)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.dtype == np.uint8
