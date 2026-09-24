"""
OCR extraction module using EasyOCR.

Wraps EasyOCR's Reader with:
  - Lazy model loading (first call only)
  - Confidence-threshold filtering
  - Structured result dataclass
  - Optional bounding-box visualisation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextRegion:
    """A single detected text region."""

    text: str
    confidence: float
    # top-left, top-right, bottom-right, bottom-left corners
    bbox: list[list[int]]

    def __str__(self) -> str:  # noqa: D105
        return f"[{self.confidence:.0%}] {self.text}"


@dataclass
class ExtractionResult:
    """Full result returned by :class:`OCRExtractor`."""

    regions: List[TextRegion]
    raw_text: str          # all regions joined with newlines
    language: str
    confidence_avg: float  # mean confidence across kept regions

    def __str__(self) -> str:
        return self.raw_text


class OCRExtractor:
    """
    Runs EasyOCR on a preprocessed image.

    Parameters
    ----------
    languages:
        List of language codes recognised by EasyOCR (default: English).
    confidence_threshold:
        Regions below this confidence are discarded (0–1).
    gpu:
        Use GPU if available.  Defaults to False for broad compatibility.
    """

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        confidence_threshold: float = 0.3,
        gpu: bool = False,
    ) -> None:
        self.languages = list(languages)
        self.threshold = confidence_threshold
        self.gpu = gpu
        self._reader = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def reader(self):
        """Lazily initialise the EasyOCR Reader (downloads models on first use)."""
        if self._reader is None:
            import easyocr  # local import to keep startup fast

            logger.info(
                "Loading EasyOCR models for languages: %s …", self.languages
            )
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR ready.")
        return self._reader

    def extract(self, image: np.ndarray) -> ExtractionResult:
        """
        Run OCR on *image* and return structured results.

        Parameters
        ----------
        image:
            Preprocessed grayscale or BGR image (NumPy array).
        """
        raw = self.reader.readtext(image, detail=1, paragraph=False)
        regions: list[TextRegion] = []

        for bbox, text, conf in raw:
            text = text.strip()
            if not text or conf < self.threshold:
                continue
            regions.append(
                TextRegion(
                    text=text,
                    confidence=float(conf),
                    bbox=[list(map(int, pt)) for pt in bbox],
                )
            )

        raw_text = "\n".join(r.text for r in regions)
        avg_conf = (
            sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        )

        return ExtractionResult(
            regions=regions,
            raw_text=raw_text,
            language=", ".join(self.languages),
            confidence_avg=avg_conf,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def annotate(
        self,
        image: np.ndarray,
        result: ExtractionResult,
        out_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on *image*.

        Parameters
        ----------
        image:
            Original (colour) image.
        result:
            ExtractionResult from :meth:`extract`.
        out_path:
            If given, saves the annotated image to this path.

        Returns
        -------
        np.ndarray
            Annotated BGR image.
        """
        annotated = image.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for region in result.regions:
            pts = np.array(region.bbox, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            label = f"{region.text[:30]} ({region.confidence:.0%})"
            cv2.putText(
                annotated, label, (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1,
                cv2.LINE_AA,
            )

        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)
            logger.info("Annotated image saved → %s", out_path)

        return annotated
