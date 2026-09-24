"""
Generate a synthetic handwriting-style sample image for testing.

Uses a cursive/italic TrueType font to simulate handwritten notes.
For the most realistic test, replace with a real photo of handwriting.

Run:
    python scripts/generate_handwriting_sample.py
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUTPUT = Path("samples/handwriting_sample.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        # Windows italic fonts (closer to handwriting)
        r"C:\Windows\Fonts\timesi.ttf",    # Times New Roman Italic
        r"C:\Windows\Fonts\calibrii.ttf",  # Calibri Italic
        r"C:\Windows\Fonts\ariali.ttf",    # Arial Italic
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        # Linux / macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ── Build image ───────────────────────────────────────────────────────────────
W, H = 900, 520
img = Image.new("RGB", (W, H), color=(245, 240, 228))  # off-white paper
draw = ImageDraw.Draw(img)

# Faint ruled lines
for y in range(70, H, 55):
    draw.line([(30, y), (W - 30, y)], fill=(190, 200, 215), width=1)

font_heading = _load_font(30)
font_body    = _load_font(26)

lines = [
    (font_heading, "Meeting Notes  —  7 September 2026", (10, 20, 90)),
    (font_body,    "Action items:",                       (5,  5,  5)),
    (font_body,    "  1.  Review OCR pipeline output",    (10, 10, 80)),
    (font_body,    "  2.  Test on real handwritten scans", (10, 10, 80)),
    (font_body,    "  3.  Tune confidence threshold",     (10, 10, 80)),
    (font_body,    "Total budget:   $4,200",              (5,  5,  5)),
    (font_body,    "Signed:  A. Developer",               (60, 10, 10)),
]

y = 28
for font, text, color in lines:
    x = random.randint(38, 52)
    dy = random.randint(-2, 2)
    draw.text((x, y + dy), text, fill=color, font=font)
    y += 58

# Slight rotation to simulate un-straight writing
img = img.rotate(random.uniform(-1.2, 1.2), fillcolor=(245, 240, 228), expand=False)

# Subtle paper noise
arr = np.array(img).astype(np.int16)
noise = np.random.randint(-6, 6, arr.shape, dtype=np.int16)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
img = Image.fromarray(arr)

img.save(OUTPUT)
print(f"Handwriting sample saved → {OUTPUT}")
print("Run: python main.py samples/handwriting_sample.png --handwriting")
