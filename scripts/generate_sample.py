"""
Generate a synthetic printed-text sample image for testing the pipeline.

Uses a real TrueType font (Arial on Windows, DejaVu on Linux/Mac) so that
EasyOCR can read it accurately — PIL's built-in bitmap font is far too small.

Run:
    python scripts/generate_sample.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path("samples/sample.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── Font resolution (real TTF required for EasyOCR accuracy) ─────────────────
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        # Windows
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        # Linux / macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    # Last resort — still better than nothing
    return ImageFont.load_default()


# ── Build image ───────────────────────────────────────────────────────────────
W, H = 900, 420
img = Image.new("RGB", (W, H), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

font_title  = _load_font(36)
font_normal = _load_font(28)
font_small  = _load_font(22)

sections = [
    (font_title,  (40, 30),  "PixelScript OCR Pipeline",    (20, 20, 20)),
    (font_small,  (40, 90),  "-" * 48,                      (160, 160, 160)),
    (font_normal, (40, 115), "Invoice No:   INV-2024-00123", (30, 30, 30)),
    (font_normal, (40, 160), "Date:         07 September 2026", (30, 30, 30)),
    (font_normal, (40, 205), "Amount Due:   $1,234.56",      (30, 30, 30)),
    (font_small,  (40, 255), "-" * 48,                      (160, 160, 160)),
    (font_normal, (40, 278), "Bill To:      Acme Corp",      (30, 30, 30)),
    (font_normal, (40, 323), "Status:       UNPAID",         (180, 30, 30)),
    (font_small,  (40, 375), "Thank you for your business!", (80, 80, 80)),
]

for font, pos, text, color in sections:
    draw.text(pos, text, fill=color, font=font)

# Slight tilt to test deskewing
img = img.rotate(-1.5, fillcolor=(255, 255, 255), expand=False)

img.save(OUTPUT)
print(f"Sample image saved → {OUTPUT}")
