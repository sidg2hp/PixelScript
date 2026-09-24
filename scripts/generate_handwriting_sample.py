"""
Generate a synthetic handwriting-style sample image for testing.

Uses PIL's default font at varying angles/sizes to simulate handwritten notes.
For a realistic test, replace this with an actual photograph of handwriting.

Run:
    python scripts/generate_handwriting_sample.py
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUTPUT = Path("samples/handwriting_sample.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 900, 500
img = Image.new("RGB", (W, H), color=(245, 240, 228))  # off-white, like paper
draw = ImageDraw.Draw(img)

# Draw faint ruled lines (like a notepad)
for y in range(60, H, 50):
    draw.line([(30, y), (W - 30, y)], fill=(180, 190, 210), width=1)

font = ImageFont.load_default()

lines = [
    "Meeting notes - 7th September 2026",
    "Action items:",
    "  1. Review OCR pipeline output",
    "  2. Test on real handwritten scans",
    "  3. Tune confidence threshold",
    "Total budget: $4,200",
    "Signed: A. Developer",
]

y = 30
for i, line in enumerate(lines):
    # Vary position slightly to simulate handwriting irregularity
    x = random.randint(35, 55)
    dy = random.randint(-2, 2)
    # Alternate between dark blue and black ink
    color = (10, 30, 100) if i % 3 != 0 else (5, 5, 5)
    draw.text((x, y + dy), line, fill=color, font=font)
    y += 50

# Slight rotation to simulate un-straight writing
img = img.rotate(random.uniform(-1.5, 1.5), fillcolor=(245, 240, 228), expand=False)

# Add very light noise texture
import numpy as np
arr = np.array(img).astype(np.int16)
noise = np.random.randint(-8, 8, arr.shape, dtype=np.int16)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
img = Image.fromarray(arr)

img.save(OUTPUT)
print(f"Handwriting sample saved → {OUTPUT}")
print("Run the pipeline with:")
print(f"  python main.py {OUTPUT} --handwriting")
