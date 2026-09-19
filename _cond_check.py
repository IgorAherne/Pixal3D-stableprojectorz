"""Scratch diagnostic: dump the DINOv2 conditioning view LATO.2 actually sees.

Reproduces dataset.voxel_dataset.VoxelVertexDataset's render path exactly, with
our nvdiffrast patch applied, so we can eyeball whether the replacement renderer
is producing a sane white-model view. Delete when done.
"""
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
LATO = os.path.join(CODE, "third_party", "lato2")
sys.path.insert(0, LATO)
os.chdir(LATO)
sys.path.insert(0, CODE)

import lato2_render_patch

lato2_render_patch.apply()

import numpy as np
from PIL import Image

from dataset.voxel_dataset import VoxelVertexDataset

SRC = sys.argv[1]
DST = sys.argv[2]

ds = VoxelVertexDataset(root_dir=SRC, render=True, need_encoder_inputs=False)
for i in range(len(ds)):
    d = ds[i]
    if "error" in d:
        print(f"{d['name']:<12} ERROR {d['error'][:400]}")
        continue
    img = np.asarray(d["image"])
    out = os.path.join(DST, f"cond_{d['name']}.png")
    Image.fromarray(img).save(out)
    cover = (img.sum(-1) > 10).mean() * 100
    print(
        f"{d['name']:<12} {img.shape} {img.dtype} "
        f"min={img.min()} max={img.max()} mean={img.mean():.1f} "
        f"object_coverage={cover:.1f}%  -> {out}"
    )
