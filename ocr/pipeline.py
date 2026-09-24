"""
High-level OCR pipeline.

Ties together the Preprocessor and OCRExtractor into a single callable,
loading images from file paths or raw NumPy arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .extractor import ExtractionResult, OCRExtractor
from .handwriting import HandwritingExtractor, HandwritingResult
from .preprocessor import PreprocessConfig, Preprocessor

logger = logging.getLogger(__name__)


class OCRPipeline:
    """
    End-to-end OCR pipeline.

    Usage
    -----
    >>> pipeline = OCRPipeline()
    >>> result = pipeline.run("samples/invoice.png")
    >>> print(result.raw_text)

    # Handwriting mode (uses TrOCR instead of EasyOCR):
    >>> pipeline = OCRPipeline(handwriting=True)
    >>> result = pipeline.run("samples/note.png")
    >>> print(result.raw_text)

    Parameters
    ----------
    languages:
        EasyOCR language codes (ignored when handwriting=True).
    confidence_threshold:
        Discard EasyOCR detections below this confidence (0–1).
    gpu:
        Forward to the OCR engine's GPU flag.
    preprocess_config:
        Fine-tune preprocessing behaviour.  Pass ``None`` for defaults.
    save_annotated:
        Directory to write annotated images to.  Disabled when ``None``.
    handwriting:
        When True, uses Microsoft TrOCR (transformer-based model fine-tuned
        on the IAM handwriting dataset) instead of EasyOCR, and switches the
        preprocessor to CLAHE-based contrast enhancement instead of hard
        binarization.  Best for cursive / mixed handwriting.
    trocr_model:
        HuggingFace model ID for TrOCR.  Defaults to the base handwritten
        variant.  Use ``"microsoft/trocr-large-handwritten"`` for higher
        accuracy at the cost of speed.
    """

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        confidence_threshold: float = 0.3,
        gpu: bool = False,
        preprocess_config: PreprocessConfig | None = None,
        save_annotated: str | Path | None = "results",
        handwriting: bool = False,
        trocr_model: str = "microsoft/trocr-base-handwritten",
    ) -> None:
        self.handwriting = handwriting

        # Auto-enable handwriting preprocessing mode if not explicitly configured
        if preprocess_config is None:
            preprocess_config = PreprocessConfig(handwriting_mode=handwriting)
        elif handwriting and not preprocess_config.handwriting_mode:
            logger.warning(
                "handwriting=True but PreprocessConfig.handwriting_mode=False. "
                "Consider setting handwriting_mode=True in your PreprocessConfig."
            )

        self.preprocessor = Preprocessor(preprocess_config)

        if handwriting:
            self.extractor = HandwritingExtractor(model_id=trocr_model, gpu=gpu)
        else:
            self.extractor = OCRExtractor(
                languages=languages,
                confidence_threshold=confidence_threshold,
                gpu=gpu,
            )
        self.save_annotated = Path(save_annotated) if save_annotated else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source: str | Path | np.ndarray,
    ) -> ExtractionResult | HandwritingResult:
        """
        Process *source* through the full OCR pipeline.

        Parameters
        ----------
        source:
            File path (str / Path) or a BGR NumPy array.

        Returns
        -------
        ExtractionResult  (printed mode)
        HandwritingResult (handwriting mode)
        """
        original, stem = self._load(source)

        # ── Preprocessing ───────────────────────────────────────────────
        logger.info(
            "Preprocessing image (mode: %s) …",
            "handwriting" if self.handwriting else "printed",
        )
        preprocessed = self.preprocessor.process(original)

        # ── Extraction ──────────────────────────────────────────────────
        logger.info("Running OCR …")
        result = self.extractor.extract(preprocessed)

        if self.handwriting:
            logger.info("Extracted %d line(s).", len(result.regions))
        else:
            logger.info(
                "Extracted %d regions (avg confidence %.1f%%).",
                len(result.regions),
                result.confidence_avg * 100,
            )

        # ── Annotated output ────────────────────────────────────────────
        if self.save_annotated and result.regions:
            out_path = self.save_annotated / f"{stem}_annotated.png"
            self.extractor.annotate(original, result, out_path=out_path)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(
        self, source: str | Path | np.ndarray
    ) -> tuple[np.ndarray, str]:
        """Load image and return (bgr_array, stem)."""
        if isinstance(source, np.ndarray):
            return source, "array"
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2.imread could not decode: {path}")
        logger.info("Loaded '%s' — %dx%d px", path.name, img.shape[1], img.shape[0])
        return img, path.stem
