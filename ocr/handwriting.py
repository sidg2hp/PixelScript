"""
Handwriting OCR module using Microsoft TrOCR.

TrOCR (Transformer-based OCR) is a vision-encoder + language-model architecture
fine-tuned specifically on handwritten text (IAM dataset).  It consistently
outperforms general-purpose OCR engines on cursive and mixed handwriting.

Model variants (downloaded from HuggingFace on first use, ~400 MB each):
  - "microsoft/trocr-base-handwritten"   ← default, best speed/accuracy trade-off
  - "microsoft/trocr-large-handwritten"  ← higher accuracy, slower
  - "microsoft/trocr-base-printed"       ← use this for printed/typed text instead

Reference: https://huggingface.co/microsoft/trocr-base-handwritten
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default HuggingFace model ID
_DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


@dataclass(frozen=True)
class HandwritingRegion:
    """One detected line / word region with its transcription."""

    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass
class HandwritingResult:
    """Result from :class:`HandwritingExtractor`."""

    regions: List[HandwritingRegion]
    raw_text: str
    model: str

    def __str__(self) -> str:
        return self.raw_text


class HandwritingExtractor:
    """
    Segment a handwritten image into text lines, then run TrOCR on each.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.
    gpu:
        Use CUDA if available.
    min_line_height:
        Ignore contour-based regions shorter than this many pixels (noise filter).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        gpu: bool = False,
        min_line_height: int = 10,
    ) -> None:
        self.model_id = model_id
        self.gpu = gpu
        self.min_line_height = min_line_height
        self._processor = None
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._processor is not None:
            return
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("Loading TrOCR model '%s' …", self.model_id)
        self._processor = TrOCRProcessor.from_pretrained(self.model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)

        if self.gpu:
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("TrOCR running on GPU.")
            else:
                logger.warning("GPU requested but CUDA not available — using CPU.")
        self._model.eval()
        logger.info("TrOCR ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image: np.ndarray) -> HandwritingResult:
        """
        Extract handwritten text from *image*.

        The image is segmented into horizontal text-line strips using
        connected-component analysis, then each strip is passed to TrOCR.
        Falls back to passing the whole image as one region if segmentation
        yields nothing.

        Parameters
        ----------
        image:
            BGR or grayscale NumPy array.

        Returns
        -------
        HandwritingResult
        """
        self._load()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        regions = self._segment_lines(gray)

        if not regions:
            logger.warning("Line segmentation found nothing — using whole image.")
            h, w = gray.shape
            regions = [(0, 0, w, h)]

        logger.info("Running TrOCR on %d region(s) …", len(regions))
        hw_regions: list[HandwritingRegion] = []
        for bbox in regions:
            x, y, w, h = bbox
            crop = image[y : y + h, x : x + w]
            text = self._run_trocr(crop)
            if text:
                hw_regions.append(HandwritingRegion(text=text, bbox=bbox))

        raw_text = "\n".join(r.text for r in hw_regions)
        return HandwritingResult(
            regions=hw_regions,
            raw_text=raw_text,
            model=self.model_id,
        )

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _segment_lines(
        self, gray: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Segment the image into horizontal text-line bounding boxes using
        morphological operations + connected components.

        Returns list of (x, y, w, h) tuples sorted top-to-bottom.
        """
        # Binarise (invert so text = white for morphology)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate horizontally to merge characters into word/line blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours of the merged blobs
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        img_h, img_w = gray.shape
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < self.min_line_height or w < 20:
                continue
            # Add small vertical padding so ascenders/descenders aren't clipped
            pad = max(4, h // 8)
            y0 = max(0, y - pad)
            y1 = min(img_h, y + h + pad)
            x0 = max(0, x - 4)
            x1 = min(img_w, x + w + 4)
            boxes.append((x0, y0, x1 - x0, y1 - y0))

        # Sort top-to-bottom (reading order)
        boxes.sort(key=lambda b: b[1])
        return boxes

    # ------------------------------------------------------------------
    # TrOCR inference
    # ------------------------------------------------------------------

    def _run_trocr(self, crop: np.ndarray) -> str:
        """Run TrOCR on a single image crop and return the decoded string."""
        import torch

        # Convert BGR → RGB PIL image
        if crop.ndim == 2:
            pil = Image.fromarray(crop).convert("RGB")
        else:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        pixel_values = self._processor(images=pil, return_tensors="pt").pixel_values
        if self.gpu and next(self._model.parameters()).is_cuda:
            pixel_values = pixel_values.cuda()

        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def annotate(
        self,
        image: np.ndarray,
        result: HandwritingResult,
        out_path: str | Path | None = None,
    ) -> np.ndarray:
        """Draw region boxes and transcription labels on *image*."""
        annotated = image.copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for region in result.regions:
            x, y, w, h = region.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 100, 0), 2)
            label = region.text[:50]
            cv2.putText(
                annotated, label, (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 80, 0), 1,
                cv2.LINE_AA,
            )

        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)
            logger.info("Annotated image saved → %s", out_path)

        return annotated
