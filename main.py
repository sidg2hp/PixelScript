"""
PixelScript — CLI entry point.

Usage examples
--------------
# Basic (English, default settings):
    python main.py samples/receipt.png

# Handwritten text (uses TrOCR):
    python main.py samples/note.png --handwriting

# Handwriting + larger model for better accuracy:
    python main.py samples/note.png --handwriting --trocr-model microsoft/trocr-large-handwritten

# Multiple languages:
    python main.py samples/document.jpg --lang en fr de

# High-confidence only, save debug pre-processing images:
    python main.py samples/scan.png --confidence 0.6 --debug

# JSON output:
    python main.py samples/form.png --json

# Disable annotated-image saving:
    python main.py samples/photo.png --no-annotate
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

from ocr.pipeline import OCRPipeline
from ocr.preprocessor import PreprocessConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smarteye",
        description="OCR pipeline powered by OpenCV + EasyOCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("image", type=Path, help="Path to input image")
    p.add_argument(
        "--lang", nargs="+", default=["en"], metavar="LANG",
        help="EasyOCR language codes (e.g. en fr de ch_sim)",
    )
    p.add_argument(
        "--confidence", type=float, default=0.3, metavar="THRESH",
        help="Minimum detection confidence [0–1]",
    )
    p.add_argument("--gpu", action="store_true", help="Use GPU if available")
    p.add_argument(
        "--no-annotate", dest="annotate", action="store_false",
        help="Skip saving the annotated output image",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Save per-step preprocessing images to results/debug/",
    )
    p.add_argument(
        "--binarize", choices=["adaptive", "otsu"], default="adaptive",
        help="Binarization algorithm",
    )
    p.add_argument(
        "--no-deskew", dest="deskew", action="store_false",
        help="Disable automatic deskewing",
    )
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Emit structured JSON instead of plain text")
    p.add_argument(
        "--handwriting", action="store_true",
        help="Use TrOCR (transformer model) optimised for handwritten text",
    )
    p.add_argument(
        "--trocr-model", default="microsoft/trocr-base-handwritten",
        metavar="MODEL_ID",
        help="HuggingFace TrOCR model ID (only used with --handwriting)",
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Force UTF-8 on Windows so box-drawing / arrow chars don't crash cp1252
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pre_cfg = PreprocessConfig(
        binarize_method=args.binarize,
        deskew=args.deskew,
        save_debug=args.debug,
        handwriting_mode=args.handwriting,
    )

    pipeline = OCRPipeline(
        languages=args.lang,
        confidence_threshold=args.confidence,
        gpu=args.gpu,
        preprocess_config=pre_cfg,
        save_annotated="results" if args.annotate else None,
        handwriting=args.handwriting,
        trocr_model=args.trocr_model,
    )

    try:
        result = pipeline.run(args.image)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        if args.handwriting:
            output = {
                "mode": "handwriting",
                "model": result.model,
                "regions": [
                    {"text": r.text, "bbox": list(r.bbox)}
                    for r in result.regions
                ],
                "raw_text": result.raw_text,
            }
        else:
            output = {
                "mode": "printed",
                "language": result.language,
                "confidence_avg": round(result.confidence_avg, 4),
                "regions": [
                    {
                        "text": r.text,
                        "confidence": round(r.confidence, 4),
                        "bbox": r.bbox,
                    }
                    for r in result.regions
                ],
                "raw_text": result.raw_text,
            }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if not result.raw_text:
            print("(no text detected)")
        else:
            sep = "-" * 60
            print(f"\n{sep}")
            print(result.raw_text)
            print(sep)
            if args.handwriting:
                print(f"\n{len(result.regions)} line(s) detected via TrOCR")
            else:
                print(
                    f"\n{len(result.regions)} region(s) | "
                    f"avg confidence {result.confidence_avg:.1%}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
