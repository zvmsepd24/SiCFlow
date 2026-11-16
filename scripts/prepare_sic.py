"""Downsample the stitched, mount-free image of the SiC/SiC section to 1.04 um/px.

Source: the stitched section with mounting material removed (0.52 um/px, grayscale stored as RGBA
with identical channels). Twofold Lanczos downsampling gives fiber diameters of about 14 px. A narrow
stitching border on the left edge is trimmed. Pores inside the composite are kept.
Usage: python scripts/prepare_sic.py PATH_TO_STITCHED_TIFF   -> data/sic/section_plane.npy
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/sic"
LEFT_TRIM = 48  # full-resolution pixels


def main(src):
    a = np.asarray(Image.open(src))[..., 0][:, LEFT_TRIM:]
    h, w = a.shape[0] // 2, a.shape[1] // 2
    plane = np.asarray(Image.fromarray(a).resize((w, h), Image.Resampling.LANCZOS))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "section_plane.npy", plane)
    print(plane.shape, plane.dtype, plane.mean())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
