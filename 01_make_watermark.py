import numpy as np
from PIL import Image, ImageDraw
import os

OUT_DIR = "data/watermarks"
os.makedirs(OUT_DIR, exist_ok=True)

# Create binary watermark logo
size = 64
img = Image.new("L", (size, size), color=0)
draw = ImageDraw.Draw(img)

# simple visible watermark pattern (customize if you want)
draw.rectangle([4, 4, size-4, size-4], outline=255, width=2)
draw.text((12, 22), "KV", fill=255)  # change initials if you want

wm_path = os.path.join(OUT_DIR, "wm_logo.png")
img.save(wm_path)

print("Saved watermark logo at:", wm_path)