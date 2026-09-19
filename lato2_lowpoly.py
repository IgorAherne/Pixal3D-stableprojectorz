# lato2_lowpoly.py
# File: lato2_lowpoly.py
"""
Retopologise a generated Pixal3D mesh into a low-poly one with LATO.2.

LATO.2 factorises mesh generation into a vertex flow (V-Flow, which places a
controllable number of vertices) and a topology flow (T-Flow, which predicts the
connectivity between them), conditioned on a rendered view encoded by DINOv2.

It lives in third_party/lato2 - a git submodule, or the same files unpacked from
the release zip - and is driven as a subprocess rather than imported, for two
reasons:

  * LATO.2 owns the top-level package names `models`, `modules`, `utils` and
    `dataset`. Putting its folder on sys.path inside the Pixal3D process would
    shadow anything else answering to those names.
  * Its ~3.6 GB of weights are released the moment the process exits, which
    matters when the Pixal3D pipeline has just finished on the same GPU.

Two things worth knowing about the output:

  * LATO.2 emits vertices and faces only - no UVs, no materials. The low-poly
    mesh does not inherit the high-poly's PBR texture.
  * It works in a normalised [-0.5, 0.5] box. `run_lato2` maps the result back
    onto the input mesh's own frame so the two line up when opened together.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

CODE_DIR = Path(__file__).parent.resolve()
LATO2_DIR = CODE_DIR / "third_party" / "lato2"
LATO2_SCRIPT = LATO2_DIR / "scripts" / "e2e_inference.py"
# Wrapper that applies the Windows render patch, then runs the script above.
LATO2_RUNNER = CODE_DIR / "lato2_run.py"
LATO2_CKPT_DIR = CODE_DIR / "MODELS" / "lato2"

CKPT_FILES = {
    "vflow_ckpt": "vflow.pt",
    "vvae_ckpt": "vvae.pt",
    "offset_head_ckpt": "offset_head.pt",
    "tflow_ckpt": "tflow.pt",
    "tvae_ckpt": "tvae.pt",
    "voxel_encoder_ckpt": "voxel_encoder.pt",
}

DEFAULT_VERT_NUM = 2000

# LATO.2's own example meshes are ~9k verts / ~11k faces, and its topology flow
# behaves badly well outside that range. A raw Pixal3D export is ~650k verts /
# ~920k faces, and feeding that in unchanged makes the edge predictor massively
# over-connect: measured on one asset, 6.66 faces per vertex (a closed manifold
# has ~2) with 81% of edges shared by more than two faces. Decimating the input
# to roughly LATO.2's own scale first brought that to 2.04 and 36%.
DEFAULT_SIMPLIFY_FACES = 12000


class Lato2Unavailable(RuntimeError):
    """LATO.2 source or checkpoints are missing."""


def lato2_available() -> bool:
    """True when both the source and every checkpoint are in place."""
    return LATO2_SCRIPT.exists() and all(
        (LATO2_CKPT_DIR / name).exists() for name in CKPT_FILES.values()
    )


def _check_available():
    if not LATO2_SCRIPT.exists():
        raise Lato2Unavailable(
            f"LATO.2 source not found at {LATO2_DIR}.\n"
            "Run install.py, or `git submodule update --init third_party/lato2`."
        )
    missing = [n for n in CKPT_FILES.values() if not (LATO2_CKPT_DIR / n).exists()]
    if missing:
        raise Lato2Unavailable(
            f"LATO.2 checkpoints missing from {LATO2_CKPT_DIR}: {missing}\n"
            "Run install.py to download them."
        )


def _input_frame(mesh_path: str):
    """Recover the centre/extent LATO.2 normalises the input mesh by.

    Mirrors quantize_mesh_clustering in dataset/utils.py: vertices are mapped to
    (v - center) / max_extent, and the exported mesh stays in that space. Must be
    computed from the file actually handed to LATO.2, decimation included.
    """
    import numpy as np
    import trimesh

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    max_extent = max(float((bbox_max - bbox_min).max()), 1e-7)
    return center, max_extent


def _decimate(src: str, dst: str, target_faces: int) -> bool:
    """Decimate `src` to ~target_faces with CuMesh. False if it wasn't needed."""
    import numpy as np
    import torch
    import trimesh
    import cumesh

    mesh = trimesh.load(src, process=False, force="mesh")

    # A textured GLB splits vertices along UV and normal seams, and trimesh keeps
    # them apart by default. Pixal3D's export loads as 647k verts with 351k
    # boundary edges that way; welding on position alone gives 459k verts and a
    # closed surface. Decimating the unwelded version shatters it.
    before = len(mesh.vertices)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    if len(mesh.vertices) != before:
        print(f"[LATO.2] Welded seam vertices {before} -> {len(mesh.vertices)}")

    if len(mesh.faces) <= target_faces:
        return False

    cm = cumesh.CuMesh()
    cm.init(
        torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda"),
        torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32, device="cuda"),
    )
    cm.simplify(int(target_faces))
    verts, faces = cm.read()

    out = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy(), process=False)
    # simplify() leaves the collapsed vertices in place; they would skew the
    # bounding box LATO.2 normalises by.
    out.remove_unreferenced_vertices()
    out.export(dst)
    print(f"[LATO.2] Decimated input {len(mesh.faces)} -> {len(out.faces)} faces for retopology")
    return True


def run_lato2(
    input_mesh: str,
    output_path: str,
    vert_num: int = DEFAULT_VERT_NUM,
    vflow_steps: int = 24,
    tflow_steps: int = 50,
    cfg_strength: float = 3.0,
    edge_threshold: float = 0.0,
    simplify_faces: int = DEFAULT_SIMPLIFY_FACES,
    fix_winding: bool = True,
    double_sided: bool = False,
    seed: int = 42,
    keep_input_frame: bool = True,
    timeout: Optional[int] = 1800,
) -> str:
    """Generate a low-poly mesh from `input_mesh` and write it to `output_path`.

    Args:
        input_mesh: High-poly mesh (.glb/.obj/.ply/.stl/.gltf/.off).
        output_path: Where to write the low-poly mesh. The extension decides the
            format (trimesh handles .glb/.obj/.ply).
        vert_num: Target vertex count for V-Flow, clamped by LATO.2 to [200, 5000].
        vflow_steps / tflow_steps: Euler steps for the two flows.
        cfg_strength: Classifier-free guidance on the rendered view condition.
        edge_threshold: Logit cutoff for the topology flow's edge predictor
            (`logits > threshold`, so 0.0 means p > 0.5). Raise it to prune
            low-confidence edges, which cuts spurious overlapping triangles at
            the cost of more holes.
        simplify_faces: Decimate the input to about this many faces first; 0
            disables. LATO.2 is tuned for ~11k-face inputs.
        fix_winding: Attempt to make face orientation consistent afterwards.
        double_sided: Write the GLB material with `doubleSided: true`. Off by
            default so you see the mesh as it really is. Turn it on to hide the
            see-through holes that backface culling exposes - it is a display
            workaround, not a repair. See the comment at the call site.
        keep_input_frame: Map the result back onto the input mesh's position and
            scale. Turn off to get LATO.2's normalised [-0.5, 0.5] output.
        timeout: Seconds before the subprocess is killed; None to wait forever.
    """
    _check_available()

    import trimesh

    input_mesh = str(Path(input_mesh).resolve())
    stem = Path(input_mesh).stem

    work_dir = Path(tempfile.mkdtemp(prefix="pixal3d_lato2_"))
    mesh_dir = work_dir / "in"
    out_dir = work_dir / "out"
    mesh_dir.mkdir()

    try:
        # LATO.2 takes a directory, so hand it one holding just this mesh.
        staged = mesh_dir / Path(input_mesh).name
        if not (simplify_faces and _decimate(input_mesh, str(staged), simplify_faces)):
            shutil.copy2(input_mesh, staged)

        cmd = [
            sys.executable, str(LATO2_RUNNER),
            "--mesh_dir", str(mesh_dir),
            "--out_dir", str(out_dir),
            "--vert_num", str(vert_num),
            "--vflow_steps", str(vflow_steps),
            "--tflow_steps", str(tflow_steps),
            "--cfg_strength", str(cfg_strength),
            "--edge_threshold", str(edge_threshold),
            "--seed", str(seed),
            # Each DataLoader worker builds its own Open3D offscreen renderer for
            # the conditioning view; on Windows that is a reliable way to hang.
            "--num_workers", "0",
            "--batch_size", "1",
            "--dino_hub_dir", str(LATO2_CKPT_DIR / "dinov2"),
        ]
        for flag, name in CKPT_FILES.items():
            cmd += [f"--{flag}", str(LATO2_CKPT_DIR / name)]

        print(f"[LATO.2] Retopologising {input_mesh} (vert_num={vert_num})...")
        result = subprocess.run(cmd, cwd=str(LATO2_DIR), timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"LATO.2 exited with code {result.returncode}")

        produced = out_dir / f"{stem}_pred.obj"
        if not produced.exists():
            raise RuntimeError(
                f"LATO.2 produced no mesh at {produced}. "
                "Too few generated vertices, or no faces decoded - see the log above."
            )

        tri = trimesh.load(str(produced), process=False, force="mesh")
        if keep_input_frame:
            # Frame comes from the staged file, which is what LATO.2 normalised.
            center, max_extent = _input_frame(str(staged))
            tri.vertices = tri.vertices * max_extent + center
        if fix_winding:
            trimesh.repair.fix_normals(tri)

        if double_sided:
            # LATO.2 tiles the surface with overlapping triangles: measured on one
            # asset, 73% of edges are shared by more than two faces, yet p99 of
            # those faces sit within 0.7% of the bounding diagonal of the input
            # surface - they are not stray geometry, the surface is genuinely
            # multi-layered. Such a mesh has no consistent orientation to find
            # (fix_normals reports success but leaves 50% of faces pointing
            # inward), so the fix that actually works is to stop culling it.
            from trimesh.visual.material import PBRMaterial

            tri.visual = trimesh.visual.TextureVisuals(
                material=PBRMaterial(
                    name="lowpoly",
                    baseColorFactor=[220, 220, 225, 255],
                    metallicFactor=0.0,
                    roughnessFactor=0.9,
                    doubleSided=True,
                )
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        tri.export(output_path)
        print(f"[LATO.2] Low-poly saved to: {output_path} "
              f"({len(tri.vertices)} verts, {len(tri.faces)} faces)")
        return output_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LATO.2 retopology for a Pixal3D mesh")
    parser.add_argument("--input", required=True, help="High-poly mesh to retopologise")
    parser.add_argument("--output", default=None, help="Output mesh (default: <input>_lowpoly.glb)")
    parser.add_argument("--vert_num", type=int, default=DEFAULT_VERT_NUM,
                        help=f"Target vertex count, clamped to [200, 5000] (default: {DEFAULT_VERT_NUM})")
    parser.add_argument("--vflow_steps", type=int, default=24)
    parser.add_argument("--tflow_steps", type=int, default=50)
    parser.add_argument("--cfg_strength", type=float, default=3.0)
    parser.add_argument("--edge_threshold", type=float, default=0.0,
                        help="Edge-predictor logit cutoff; raise to prune spurious faces.")
    parser.add_argument("--simplify_faces", type=int, default=DEFAULT_SIMPLIFY_FACES,
                        help=f"Decimate the input to ~N faces first, 0 to disable "
                             f"(default: {DEFAULT_SIMPLIFY_FACES}).")
    parser.add_argument("--no_fix_winding", action="store_true",
                        help="Leave LATO.2's inconsistent face orientation as-is.")
    parser.add_argument("--double_sided", action="store_true",
                        help="Write a doubleSided material to hide the see-through holes. "
                             "A display workaround, not a repair - the mesh stays "
                             "non-orientable either way.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw_frame", action="store_true",
                        help="Leave the output in LATO.2's normalised [-0.5, 0.5] space.")
    args = parser.parse_args()

    out = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_lowpoly.glb"))
    run_lato2(
        args.input, out,
        vert_num=args.vert_num,
        vflow_steps=args.vflow_steps,
        tflow_steps=args.tflow_steps,
        cfg_strength=args.cfg_strength,
        edge_threshold=args.edge_threshold,
        simplify_faces=args.simplify_faces,
        fix_winding=not args.no_fix_winding,
        double_sided=args.double_sided,
        seed=args.seed,
        keep_input_frame=not args.raw_frame,
    )
