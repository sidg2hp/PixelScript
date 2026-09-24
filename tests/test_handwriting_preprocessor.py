"""
Unit tests for handwriting-mode preprocessing.
No model downloads required — tests only the preprocessing layer.
"""

import numpy as np
import pytest

from ocr.preprocessor import Preprocessor, PreprocessConfig


def _make_image(w: int = 400, h: int = 300) -> np.ndarray:
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    # Simulate pen strokes with grey lines (not pure black — as in real handwriting)
    img[80:85, 50:350] = (60, 40, 30)
    img[140:144, 60:320] = (50, 50, 60)
    img[200:205, 40:280] = (70, 60, 50)
    return img


def test_handwriting_mode_output_is_grayscale():
    img = _make_image()
    cfg = PreprocessConfig(handwriting_mode=True, denoise=False, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    assert out.ndim == 2


def test_handwriting_mode_preserves_grey_values():
    """CLAHE output should NOT be strictly binary (0/255 only)."""
    img = _make_image()
    cfg = PreprocessConfig(handwriting_mode=True, denoise=False, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    unique = np.unique(out)
    # More than two distinct values → not hard-binarized
    assert len(unique) > 2, "Handwriting mode should preserve greyscale gradation"


def test_printed_mode_is_binary():
    """Normal (non-handwriting) mode must hard-binarize."""
    img = _make_image()
    cfg = PreprocessConfig(handwriting_mode=False, denoise=False, binarize=True, deskew=False, border_px=0)
    out = Preprocessor(cfg).process(img)
    unique = np.unique(out)
    assert set(unique).issubset({0, 255})


def test_handwriting_mode_full_pipeline_no_crash():
    img = _make_image()
    cfg = PreprocessConfig(handwriting_mode=True)
    out = Preprocessor(cfg).process(img)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 2
    assert out.dtype == np.uint8
