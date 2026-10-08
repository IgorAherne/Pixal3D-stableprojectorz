"""Builders of the four small models every prop needs beside the flow networks and the decoders: background removal
(RMBG-2.0 behind the BiRefNet wrapper), MoGe-2 (the camera of a single picture), DINOv3 ViT-L and NAF.

One implementation for the staged engine, the launcher and the checks, so that all of them hold the same modules.
Each builder constructs its model the way the reference path does (the lines are named at each one): on the CPU,
with the model's own loader, from the local folder or file given; then it moves the model to the device asked for.
What comes back is an nn.Module in eval mode with requires_grad off, and no other copy of it is kept anywhere.

    paths = default_paths(Path(".../models/pixal3d"))     # or the checkout's MODELS junction
    build = default_builders(paths)                        # {"rembg", "moge", "dino", "naf"}: (device) -> module
    dino = build["dino"]("cuda")

    rembg   Matte: `matte(rgb picture) -> the picture with its alpha set`, the call of pipelines/rembg/BiRefNet.py;
            `.model` is the RMBG-2.0 network, `.transform_image` the wrapper's transform
    moge    moge.model.v2.MoGeModel (`.infer(image)`)
    dino    transformers.DINOv3ViTModel, what DinoV3ProjFeatureExtractor keeps as `.model`
    naf     the NAF upsampler, what DinoV3ProjFeatureExtractor keeps as `.naf_model`

The process must already be set up as Pixal3D needs it (HF_HOME, offline mode, the working directory): the
launcher's configure_env does that. Nothing here reads an environment variable or looks a model up by name.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

NAMES = ("rembg", "moge", "dino", "naf")
MOGE_REPO = "models--Ruicheng--moge-2-vitl"        # app.py MOGE_MODEL_NAME, as the Hugging Face cache names it


@dataclass(frozen=True)
class ModelPaths:
    """Local folders and files of the four models."""
    rembg: str          # folder of RMBG-2.0 (config.json, birefnet.py, model.safetensors)
    moge: str           # MoGe-2's checkpoint file (model.pt)
    dino: str           # folder of DINOv3 ViT-L (config.json, model.safetensors)
    naf_repo: str       # checkout of valeoai/NAF (hubconf.py)
    naf_weights: str    # naf_release.pth


def default_paths(models_root: Path) -> ModelPaths:
    """The layout install.py leaves under the models folder (Spellcast3D's models/pixal3d, the checkout's MODELS).
    MoGe-2 lies in a Hugging Face cache: the snapshot refs/main names, as hf_hub_download picks it offline."""
    root = Path(models_root)
    moge = root / "hub" / MOGE_REPO
    try:
        revision = (moge / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    checkpoint = moge / "snapshots" / (revision or "main") / "model.pt"
    if not checkpoint.is_file():
        found = sorted((moge / "snapshots").glob("*/model.pt"))
        if found:
            checkpoint = found[-1]
    hub = root / "torch" / "hub"
    return ModelPaths(rembg=str(root / "RMBG-2.0"), moge=str(checkpoint), dino=str(root / "dinov3"),
                      naf_repo=str(hub / "valeoai_NAF_main"), naf_weights=str(hub / "checkpoints" / "naf_release.pth"))


def default_builders(paths: ModelPaths) -> dict[str, Callable[[str], nn.Module]]:
    """name -> build(device): the module on that device, eval(), requires_grad off."""
    return {"rembg": functools.partial(build_rembg, paths), "moge": functools.partial(build_moge, paths),
            "dino": functools.partial(build_dino, paths), "naf": functools.partial(build_naf, paths)}


class Matte(nn.Module):
    """The BiRefNet wrapper of pipelines/rembg/BiRefNet.py as a module, so that it can be moved, dropped and
    compared like the other three. Calling it runs the wrapper's own `__call__` (which sends the picture to
    "cuda": the module must be on the card), and it has the wrapper's attributes `model` and `transform_image`."""

    def __init__(self, model: nn.Module, transform_image: Callable[[Any], torch.Tensor]):
        super().__init__()
        self.model = model
        self.transform_image = transform_image

    def forward(self, image: Any) -> Any:
        from ..pipelines.rembg.BiRefNet import BiRefNet

        return BiRefNet.__call__(self, image)


def _present(path: str, what: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} is missing: {path}")
    return path


def build_rembg(paths: ModelPaths, device: str) -> nn.Module:
    """`rembg.BiRefNet(model_name="briaai/RMBG-2.0")` of Pixal3DImageTo3DPipeline.from_pretrained
    (pipelines/pixal3d_image_to_3d.py:123): the body of BiRefNet.__init__ (pipelines/rembg/BiRefNet.py:11-26) with
    the folder given instead of the one resolve_local_model finds."""
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        _present(paths.rembg, "the RMBG-2.0 folder"), trust_remote_code=True, local_files_only=True
    )
    model.eval()
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    matte = Matte(model, transform_image)
    matte.eval()
    matte.requires_grad_(False)
    return matte.to(device)


def build_moge(paths: ModelPaths, device: str) -> nn.Module:
    """app.py:113-117 (load_moge_model), with the checkpoint file given instead of the repository's name."""
    from moge.model.v2 import MoGeModel

    moge_model = MoGeModel.from_pretrained(_present(paths.moge, "MoGe-2's checkpoint")).to(device)
    moge_model.eval()
    moge_model.requires_grad_(False)
    return moge_model


def build_dino(paths: ModelPaths, device: str) -> nn.Module:
    """DinoV3ProjFeatureExtractor.__init__ (trainers/flow_matching/mixins/image_conditioned_proj.py:483-486).
    The class comes from transformers itself: the launcher replaces the name inside image_conditioned_proj by a
    cache that hands every caller the one instance the reference path keeps in RAM."""
    from transformers import DINOv3ViTModel

    model = DINOv3ViTModel.from_pretrained(_present(paths.dino, "the DINOv3 folder"), local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def build_naf(paths: ModelPaths, device: str) -> nn.Module:
    """DinoV3ProjFeatureExtractor._load_naf (image_conditioned_proj.py:514-523) and the checkout's hubconf.py:
    `NAF()`, then the state of naf_release.pth. The weights file is read here, from the path given, instead of
    through torch.hub's cache folder (and as tensors only: nothing else in the file is executed)."""
    naf_model = torch.hub.load(
        _present(paths.naf_repo, "the NAF checkout"), "naf", pretrained=False, device="cpu", trust_repo=True,
        source="local"
    )
    naf_model.load_state_dict(
        torch.load(_present(paths.naf_weights, "NAF's weights"), map_location="cpu", weights_only=True)
    )
    naf_model.eval()
    naf_model.requires_grad_(False)
    return naf_model.to(device)
