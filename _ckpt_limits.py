"""Scratch diagnostic: print the vertex-count limits baked into the LATO.2 weights.

`--vert_num` is fed to the V-Flow as `count / vflow_cfg["max_vertex_num"] * 1000`,
and the T-Flow has its own `max_vertices`. Those two numbers say how far above the
default 2000 we can go before leaving the training distribution. Delete when done.
"""
import os
import sys

import torch

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODELS", "lato2")

for name in ("vflow.pt", "tflow.pt", "tvae.pt"):
    path = os.path.join(CKPT, name)
    if not os.path.exists(path):
        print(f"{name}: MISSING at {path}")
        continue
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = {k: v for k, v in blob.items() if k != "state_dict" and not hasattr(v, "shape")}
    print(f"--- {name} ---")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"      {kk} = {vv}")
        else:
            print(f"  {k} = {v}")
    print()
