# PixelScript

A lightweight Python OCR pipeline built on **OpenCV** (preprocessing) and
**EasyOCR** / **TrOCR** (text recognition). Drop in any image — printed or
handwritten — and get clean extracted text back.

---

## Project layout

```
PixelScript/
├── ocr/
│   ├── __init__.py        # public API re-exports
│   ├── preprocessor.py    # OpenCV image preprocessing chain
│   ├── extractor.py       # EasyOCR wrapper + TextRegion dataclasses
│   ├── handwriting.py     # TrOCR handwriting extractor
│   └── pipeline.py        # high-level orchestration (printed + handwriting)
├── tests/
│   ├── test_preprocessor.py
│   └── test_handwriting_preprocessor.py
├── scripts/
│   ├── generate_sample.py             # synthetic printed test image
│   └── generate_handwriting_sample.py # synthetic handwriting test image
├── samples/               # put your input images here
├── results/               # annotated outputs land here
├── main.py                # CLI entry point
└── requirements.txt
```

---

## Setup

```bash
# 1. Create & activate the virtual environment
python -m venv .venv --system-site-packages
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quick start

```bash
# Generate a synthetic printed sample
python scripts/generate_sample.py
python main.py samples/sample.png

# Generate a synthetic handwriting sample and use TrOCR
python scripts/generate_handwriting_sample.py
python main.py samples/handwriting_sample.png --handwriting
```

Expected output (printed):
```
------------------------------------------------------------
Invoice No: INV-2024-00123
Date: 07 September 2026
Amount Due: $1,234.56
Thank you for your business!
------------------------------------------------------------

6 region(s) | avg confidence 85.3%
```

An annotated image (green boxes = printed, orange boxes = handwriting) is
written to `results/<stem>_annotated.png`.

---

## CLI reference

```
python main.py <image> [options]

Positional:
  image                        Path to input image

Options:
  --lang LANG [LANG ...]       EasyOCR language codes (default: en)
  --confidence THRESH          Min confidence to keep a region [0-1] (default: 0.3)
  --gpu                        Use GPU if available
  --no-annotate                Don't save the annotated output image
  --debug                      Save per-step preprocessed images to results/debug/
  --binarize {adaptive,otsu}   Binarization algorithm (default: adaptive)
  --no-deskew                  Disable automatic deskew correction
  --handwriting                Use TrOCR (transformer) for handwritten text
  --trocr-model MODEL_ID       HuggingFace TrOCR model (default: trocr-base-handwritten)
  --json                       Emit structured JSON output
  -v, --verbose                Show debug logs
```

---

## Programmatic API

```python
from ocr import OCRPipeline
from ocr.preprocessor import PreprocessConfig

# --- Printed text ---
cfg = PreprocessConfig(binarize_method="otsu", deskew=True)
pipeline = OCRPipeline(languages=["en"], confidence_threshold=0.4, preprocess_config=cfg)
result = pipeline.run("samples/invoice.png")
print(result.raw_text)
for region in result.regions:
    print(f"  [{region.confidence:.0%}] {region.text}  bbox={region.bbox}")

# --- Handwritten text (TrOCR) ---
pipeline = OCRPipeline(handwriting=True)
result = pipeline.run("samples/note.png")
print(result.raw_text)
```

---

## Preprocessing chain

| Step | What it does |
|------|--------------|
| Resize | Upscale images narrower than `min_width` (default 1000 px) via bicubic |
| Grayscale | Convert BGR → single-channel |
| Denoise | `fastNlMeansDenoising` to remove sensor/compression noise |
| Binarize | Adaptive Gaussian threshold **or** Otsu — produces clean black/white |
| Deskew | Moment-based rotation correction (up to ±45°) |
| Border | Add white padding so edge text isn't clipped |

---

## Running tests

```bash
pytest tests/ -v
```

---

## Supported languages (examples)

| Code | Language |
|------|----------|
| `en` | English |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `ch_sim` | Simplified Chinese |
| `ar` | Arabic |

Full list: https://www.jaided.ai/easyocr/

---

## Notes

- EasyOCR downloads its models (~100 MB) on **first run** to `~/.EasyOCR/`.
- TrOCR downloads its model (~400 MB) on **first run** to `~/.cache/huggingface/`.
- For scanned documents with complex layouts consider `--confidence 0.5 --binarize otsu`.
- GPU inference (`--gpu`) requires a CUDA-capable device and matching `torch+cuda` wheels.
- `transformers` must be `>=4.35,<5.0` — TrOCR's processor API is not yet compatible with v5.
