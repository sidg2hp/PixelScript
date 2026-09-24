"""
Image preprocessing module.

Applies a configurable chain of OpenCV transforms to improve OCR accuracy:
  1. Resize (upscale small images)
  2. Grayscale conversion
  3. Denoising (Non-Local Means)
  4. Adaptive thresholding / Otsu binarization
  5. Deskewing (moment-based rotation correction)
  6. Border padding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """
    Controls which preprocessing steps are applied and their parameters.

    Set ``handwriting_mode=True`` to switch to a softer chain suited for
    handwritten images:
      - Binarization is replaced by CLAHE contrast enhancement (preserves
        stroke variation that hard thresholds destroy)
      - Denoising strength is halved to preserve fine stroke detail
    Pair with ``OCRPipeline(handwriting=True)`` for best results.
    """

    # --- mode ---
    handwriting_mode: bool = False

    # --- resize ---
    min_width: int = 1000
    """Upscale image if narrower than this (preserves aspect ratio)."""

    # --- denoise ---
    denoise: bool = True
    h_luminance: int = 10
    template_window: int = 7
    search_window: int = 21

    # --- binarization ---
    binarize: bool = True
    binarize_method: Literal["adaptive", "otsu"] = "adaptive"
    adaptive_block: int = 15   # must be odd
    adaptive_c: int = 9

    # --- deskew ---
    deskew: bool = True
    deskew_max_angle: float = 45.0

    # --- CLAHE (handwriting mode) ---
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    # --- border ---
    border_px: int = 20

    # --- save debug ---
    save_debug: bool = False
    debug_dir: Path = field(default_factory=lambda: Path("results/debug"))


class Preprocessor:
    """Applies a configurable OpenCV preprocessing pipeline to an image."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.cfg = config or PreprocessConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing chain.

        Parameters
        ----------
        image:
            BGR or grayscale image as a NumPy array (e.g. from cv2.imread).

        Returns
        -------
        np.ndarray
            Preprocessed, binarized image ready for OCR.
        """
        img = image.copy()
        steps = [
            ("resize",     self._resize),
            ("grayscale",  self._to_gray),
            ("denoise",    self._denoise),
            # handwriting mode uses CLAHE instead of hard binarization
            ("enhance",    self._enhance) if self.cfg.handwriting_mode else ("binarize", self._binarize),
            ("deskew",     self._deskew),
            ("border",     self._add_border),
        ]
        for name, fn in steps:
            img = fn(img)
            if self.cfg.save_debug:
                self._save_debug(img, name)
            logger.debug("Step '%s' complete — shape %s", name, img.shape)
        return img

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w < self.cfg.min_width:
            scale = self.cfg.min_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug("Resized %dx%d → %dx%d", w, h, new_w, new_h)
        return img

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        if not self.cfg.denoise:
            return img
        # Reduce denoising strength in handwriting mode to preserve stroke detail
        h = self.cfg.h_luminance // 2 if self.cfg.handwriting_mode else self.cfg.h_luminance
        return cv2.fastNlMeansDenoising(
            img,
            h=h,
            templateWindowSize=self.cfg.template_window,
            searchWindowSize=self.cfg.search_window,
        )

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement — used in handwriting mode instead of binarization."""
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
        )
        return clahe.apply(img)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        if not self.cfg.binarize:
            return img
        if self.cfg.binarize_method == "otsu":
            _, binary = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # adaptive (default)
            binary = cv2.adaptiveThreshold(
                img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.cfg.adaptive_block,
                self.cfg.adaptive_c,
            )
        return binary

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Correct slight rotations using image moments on the text mask."""
        if not self.cfg.deskew:
            return img

        # Invert so text is white (moments work on white pixels)
        inv = cv2.bitwise_not(img)
        coords = np.column_stack(np.where(inv > 0))
        if coords.shape[0] < 10:
            return img  # not enough content to measure

        angle = cv2.minAreaRect(coords)[-1]

        # minAreaRect returns angles in [-90, 0); map to [-45, 45]
        if angle < -45:
            angle = 90 + angle

        if abs(angle) > self.cfg.deskew_max_angle:
            logger.warning("Deskew angle %.1f° exceeds limit — skipping.", angle)
            return img

        if abs(angle) < 0.5:
            return img  # trivially straight

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255
        )
        logger.debug("Deskewed by %.2f°", angle)
        return rotated

    def _add_border(self, img: np.ndarray) -> np.ndarray:
        px = self.cfg.border_px
        return cv2.copyMakeBorder(
            img, px, px, px, px, cv2.BORDER_CONSTANT, value=255
        )

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _save_debug(self, img: np.ndarray, step_name: str) -> None:
        self.cfg.debug_dir.mkdir(parents=True, exist_ok=True)
        out = self.cfg.debug_dir / f"{step_name}.png"
        cv2.imwrite(str(out), img)
