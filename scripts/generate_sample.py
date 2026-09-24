"""
Generate a synthetic sample image for testing the pipeline.

Run:
    python scripts/generate_sample.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path("samples/sample.png")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

W, H = 900, 400
img = Image.new("RGB", (W, H), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

lines = [
    "PixelScript OCR Pipeline",
    "-------------------------------",
    "Invoice No: INV-2024-00123",
    "Date: 07 September 2026",
    "Amount Due: $1,234.56",
    "Thank you for your business!",
]

# Use default PIL font (always available)
font = ImageFont.load_default()

y = 40
for line in lines:
    draw.text((60, y), line, fill=(10, 10, 10), font=font)
    y += 45

# Add a slight tilt to test deskewing
img = img.rotate(-2, fillcolor=(255, 255, 255), expand=False)

img.save(OUTPUT)
print(f"Sample image saved → {OUTPUT}")
