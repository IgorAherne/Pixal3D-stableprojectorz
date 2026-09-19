"""Scratch diagnostic: topology + orientation report, and single-sided re-export.

Usage:  python _topo.py <mesh> [<mesh> ...] [--resave-single-sided]
Delete when done.
"""
import collections
import sys

import numpy as np
import trimesh


def report(path):
    m = trimesh.load(path, force="mesh", process=False)
    V, F = len(m.vertices), len(m.faces)
    e = np.sort(m.edges_sorted, axis=1)
    _, ec = np.unique(e, axis=0, return_counts=True)
    h = collections.Counter(ec.tolist())
    tot = len(ec)
    nm = sum(c for k, c in h.items() if k > 2)

    # Which way do faces actually point, relative to the mesh centroid?
    tris = m.vertices[m.faces]
    fc = tris.mean(axis=1)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    outward = ((fc - m.centroid) * n).sum(axis=1) > 0

    ds = None
    mat = getattr(getattr(m, "visual", None), "material", None)
    if mat is not None:
        ds = getattr(mat, "doubleSided", None)

    print(f"{path}")
    print(f"   {V} verts / {F} faces   F/V {F/V:.2f}   (clean would be {2*V-4} faces)")
    print(f"   boundary edges {h.get(1,0)} ({h.get(1,0)/tot*100:.1f}%)   "
          f"nonmanifold {nm} ({nm/tot*100:.1f}%)")
    print(f"   edge histogram {dict(sorted(h.items()))}")
    print(f"   faces outward {outward.mean()*100:.1f}%  inward {(~outward).mean()*100:.1f}%  "
          f"volume {m.volume:+.5f}")
    print(f"   is_winding_consistent {m.is_winding_consistent}   doubleSided={ds}")
    return m


if __name__ == "__main__":
    resave = "--resave-single-sided" in sys.argv
    for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
        m = report(p)
        if resave:
            out = p.rsplit(".", 1)[0] + "_singlesided.glb"
            m.visual = trimesh.visual.ColorVisuals()
            m.export(out)
            print(f"   -> re-exported single-sided: {out}")
        print()
