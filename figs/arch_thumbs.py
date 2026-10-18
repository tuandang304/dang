"""Input/output thumbnails for figs/arch_diagram.tex (illustration only, no model output).

One Kvasir test frame (PraNet TestDataset) and its ground-truth mask, resized to 176x176.
Regenerate:
  uv run --no-project --with numpy --with pillow --with huggingface_hub python figs/arch_thumbs.py
"""

import io
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

HERE = Path(__file__).resolve().parent
FRAME = "cju1hp9i2xu8e0988u2dazk7m.png"
SIZE = 176


def main():
    z = zipfile.ZipFile(hf_hub_download("Zyna1121/lightpranet-polyp-data", "TestDataset.zip", repo_type="dataset"))
    img = Image.open(io.BytesIO(z.read(f"TestDataset/Kvasir/images/{FRAME}"))).convert("RGB")
    gt = Image.open(io.BytesIO(z.read(f"TestDataset/Kvasir/masks/{FRAME}"))).convert("L")
    img = img.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    gt = np.asarray(gt.resize((SIZE, SIZE), Image.Resampling.NEAREST)) > 127
    img.save(HERE / "arch_input.png")
    # mask overlay: dimmed frame, lesion tinted sky blue as in the qualitative figure
    rgb = np.asarray(img) / 255.0
    over = rgb * 0.55
    over[gt] = 0.35 * rgb[gt] + 0.65 * np.array([0.337, 0.706, 0.914])
    Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8)).save(HERE / "arch_output.png")


if __name__ == "__main__":
    main()
