"""Render low-dpi previews of selected v2 presentation slides for inspection."""
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
doc = fitz.open(ROOT / "presentation/final_presentation_v3.pdf")
print("pages:", len(doc))
out = ROOT / "presentation/generated_assets/preview_v3"
out.mkdir(exist_ok=True)
for i in [0, 3, 7]:
    pix = doc[i].get_pixmap(dpi=80)
    pix.save(out / ("slide_%02d.png" % (i + 1)))
print("previews saved")
