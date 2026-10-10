"""The image conditioning of the four stages as functions of DINOv3's tokens.

The reference pipelines ask an extractor (DinoV3ProjFeatureExtractor, one per stage) for each conditioning, and
every extractor runs DINOv3 again and fills a dense grid of features of which the stage then reads the rows of its
tokens. Here the same work is split where it can be shared and cut down:

    image = image_tensor(images, 512, device)              # the pictures as the extractors take them
    z = encode(dino, image)                                # DINOv3 once per input size, for every stage of that size
    cond_ss = cond_dense(SPECS["ss"], z, camera)           # the dense grid of the sparse-structure stage
    cond = cond_tokens(SPECS["shape_512"], z, image, naf, camera, coords, 32)      # only the rows of `coords`

`cond_tokens` projects the whole grid as the extractors do, then samples DINOv3's patch map and NAF's upsampled map
at the tokens' points only (NAF one attention head at a time: modules/naf_sliced.py), so neither the dense grid
[R^3, 2048] nor NAF's map [1024, T, T] is ever made. The values are those the extractors give: both are built from
the same functions of trainers/flow_matching/mixins/image_conditioned_proj.py.

Everything is float32 on the device of `z`; a caller that stores a conditioning casts it itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize

from ..trainers.flow_matching.mixins.image_conditioned_proj import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ProjGrid,
    ProjGridMV,
    dino_tokens,
    pictures_to_tensor,
    tokens_to_features,
    view_tokens_to_features,
)


@dataclass(frozen=True)
class Spec:
    """One stage's extractor settings: app.IMAGE_COND_CONFIGS[name], and the two numbers of the DINOv3 model it
    names that say how its tokens are laid out."""
    name: str
    grid: int                           # grid_resolution: the side of the projection grid the stage was trained at
    image_size: int                     # side of the picture DINOv3 and NAF's guide get
    use_naf: bool = False               # use_naf_upsample
    naf_target: Optional[int] = None    # naf_target_size: side of NAF's upsampled map
    model_name: str = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
    patch_size: int = 16
    registers: int = 4

    @staticmethod
    def of(extractor: nn.Module) -> "Spec":
        """The settings of a built extractor (DinoV3ProjFeatureExtractor or its multi-view subclass). A caller
        that holds one then conditions exactly as that extractor would, whatever pipeline.json gave it."""
        target = tuple(extractor.naf_target_size) if extractor.use_naf_upsample else None
        if target is not None and target[0] != target[1]:
            raise ValueError(f"NAF's target map must be square, got {target}")
        return Spec(
            name=f"{extractor.image_size} px, grid {extractor.grid_resolution}"
                 + (f", naf {target[0]}" if target else ""),
            grid=extractor.grid_resolution,
            image_size=extractor.image_size,
            use_naf=extractor.use_naf_upsample,
            naf_target=target[0] if target else None,
            model_name=extractor.model_name,
            patch_size=extractor.patch_size,
            registers=getattr(extractor.model.config, "num_register_tokens", 4),
        )


SPECS: Dict[str, Spec] = {
    "ss": Spec("ss", grid=16, image_size=512),
    "shape_512": Spec("shape_512", grid=32, image_size=512, use_naf=True, naf_target=512),
    "shape_1024": Spec("shape_1024", grid=64, image_size=1024, use_naf=True, naf_target=512),
    "tex_1024": Spec("tex_1024", grid=64, image_size=1024, use_naf=True, naf_target=1024),
}


@dataclass
class Camera:
    """The camera arguments of an extractor. Single view (Pixal3DImageTo3DPipeline): camera_angle_x [1],
    distance [1], no matrix. Multi-view (Pixal3DMVImageTo3DPipeline): camera_angle_x [1, V], distance [1, V],
    transform_matrix [1, V, 4, 4] (camera to world, view 0 the main one)."""
    camera_angle_x: torch.Tensor
    distance: torch.Tensor
    mesh_scale: float
    transform_matrix: Optional[torch.Tensor]
    views: int

    @staticmethod
    def single(camera: dict, device) -> "Camera":
        """From the three floats of a single picture's camera, as get_proj_cond_ss builds its tensors."""
        return Camera(
            camera_angle_x=torch.tensor([camera["camera_angle_x"]], device=device),
            distance=torch.tensor([camera["distance"]], device=device),
            mesh_scale=camera.get("mesh_scale", 1.0),
            transform_matrix=None,
            views=1,
        )

    @staticmethod
    def from_bundle(views: dict, device) -> "Camera":
        """From a views bundle (see Pixal3DMVImageTo3DPipeline.run_mv), which holds the same three values that
        pipeline passes its condition builders."""
        transform_matrix = views["transform_matrix"].to(device)
        return Camera(
            camera_angle_x=torch.as_tensor(views["camera_angle_x"], dtype=torch.float32, device=device),
            distance=torch.as_tensor(views["camera_distance"], dtype=torch.float32, device=device),
            mesh_scale=views["mesh_scale"],
            transform_matrix=transform_matrix,
            views=transform_matrix.shape[1],
        )


_grids: Dict[tuple, ProjGrid] = {}


def proj_grid_for(cls: type, grid_resolution: int, image_resolution: int, device) -> ProjGrid:
    """The projection grid `cls(grid_resolution, image_resolution)` on `device`, built once per process.

    The key is all four: ProjGrid and ProjGridMV take different cameras, and the image resolution enters the
    projection itself, so a grid of the right side alone would give shifted features without any error.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (cls, int(grid_resolution), int(image_resolution), str(device))
    if key not in _grids:
        _grids[key] = cls(grid_resolution=grid_resolution, image_resolution=image_resolution).to(device)
    return _grids[key]


def image_tensor(images: Union[List, dict], size: int, device) -> torch.Tensor:
    """The pictures of one prop as a batch [V, 3, size, size], float32 in 0..1, on `device`: what DINOv3 gets
    (after `encode` normalizes it) and what NAF takes as its guide.

    images: the list of PIL pictures a single-view pipeline is given (resized here, LANCZOS), or a views bundle
    (its "images": {size: [1, V, 3, size, size]}, already resized and premultiplied by alpha).
    """
    if isinstance(images, dict):
        by_size = images.get("images", images)
        if size not in by_size:
            raise KeyError(f"No view images at resolution {size}; got {sorted(by_size)}.")
        batch = by_size[size].to(device)
        return batch.reshape(-1, *batch.shape[2:])
    return pictures_to_tensor(images, size).to(device)


@torch.no_grad()
def encode(dino: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """DINOv3's tokens z [V, 1 + registers + patches, 1024] of `image_tensor`'s batch: the extractors'
    ImageNet normalization and extract_features. All views go through the model as one batch."""
    return dino_tokens(dino, normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD))


def _features(spec: Spec, z: torch.Tensor, camera: Camera, grid: int, image: Optional[torch.Tensor],
              naf: Optional[nn.Module], point_index: Optional[torch.Tensor]) -> dict:
    """An extractor's (z_global, z_proj) for one prop, from its tokens."""
    patches = spec.image_size // spec.patch_size
    if z.shape[0] != camera.views or z.shape[1] != 1 + spec.registers + patches * patches:
        raise ValueError(f"{spec.name}: expected DINOv3 tokens [{camera.views}, "
                         f"{1 + spec.registers + patches * patches}, D] of {spec.image_size} px pictures, "
                         f"got {list(z.shape)}")
    device = z.device
    target = (spec.naf_target, spec.naf_target) if naf is not None else None
    if camera.transform_matrix is None:
        z_global, z_proj = tokens_to_features(
            proj_grid_for(ProjGrid, grid, spec.image_size, device), z, spec.registers, patches,
            camera.camera_angle_x, camera.distance, torch.tensor([camera.mesh_scale], device=device),
            image_for_naf=image, naf_model=naf, naf_target_size=target, point_index=point_index,
        )
    else:
        z_global, z_proj = view_tokens_to_features(
            proj_grid_for(ProjGridMV, grid, spec.image_size, device), z, spec.registers, patches,
            camera.camera_angle_x, camera.distance,
            torch.as_tensor(camera.mesh_scale, dtype=torch.float32, device=device).reshape(-1),
            camera.transform_matrix,
            image_for_naf=image, naf_model=naf, naf_target_size=target, point_index=point_index,
        )
    return {"global": z_global, "proj": z_proj}


@torch.no_grad()
def cond_dense(spec: Spec, z: torch.Tensor, camera: Camera) -> dict:
    """The conditioning of a stage without NAF on its whole grid (the sparse-structure stage):
    {"global": [1, 5, 1024], "proj": [1, R^3, 1024]}, what get_proj_cond_ss returns as 'cond'."""
    if spec.use_naf:
        raise ValueError(f"{spec.name} upsamples with NAF: use cond_tokens")
    return _features(spec, z, camera, spec.grid, None, None, None)


@torch.no_grad()
def cond_tokens(spec: Spec, z: torch.Tensor, image: torch.Tensor, naf: nn.Module, camera: Camera,
                coords: torch.Tensor, grid: int) -> dict:
    """The conditioning of a shape or texture stage at the prop's tokens:
    {"global": [1, 5, 1024], "proj": [K, 2048]}, the 'global' and the feats of 'proj' of get_proj_cond_shape.

    z: `encode` of the stage's picture size; image: that `image_tensor` (NAF's guide); coords [K, 4] int
    (batch, x, y, z) as the stage's sparse tensor holds them, batch 0 throughout; grid: the side R of the grid
    the coords live in (spec.grid, or the cascade's resolution // 16).
    """
    if not spec.use_naf:
        raise ValueError(f"{spec.name} has no NAF branch: use cond_dense")
    xyz = coords[:, 1:4].long()
    if xyz.numel() and (int(xyz.min()) < 0 or int(xyz.max()) >= grid or int(coords[:, 0].max()) != 0
                        or int(coords[:, 0].min()) != 0):
        raise ValueError(f"{spec.name}: coords must be (0, x, y, z) with 0 <= x, y, z < {grid}")
    # ProjGrid lists its points in meshgrid(indexing='ij') order: voxel (x, y, z) is row (x * R + y) * R + z,
    # the row `z_proj.reshape(B, R, R, R, -1)[b, x, y, z]` reads.
    point_index = (xyz[:, 0] * grid + xyz[:, 1]) * grid + xyz[:, 2]
    cond = _features(spec, z, camera, grid, image, naf, point_index)
    cond["proj"] = cond["proj"][0]
    return cond


def negative(cond: dict) -> dict:
    """The unconditional twin of a conditioning: zeros of the same shapes and dtypes, on the same device."""
    return {"global": torch.zeros_like(cond["global"]), "proj": torch.zeros_like(cond["proj"])}
