"""What the staged engine and its caller exchange: the request of one prop, what the caller's hooks hand back, the
events of a run, and the three exceptions that end a prop, a batch or the process.

Nothing here imports torch: a caller that only builds requests or reads events (a web server's record keeping, a
test with a fake engine) can import this module alone.

    request = PropRequest(id="chair", seed=11, resolution=1024, multi=False,
                          sampler_params={"ss": {...}, "shape": {...}, "tex": {...}})
    results = engine.run([request], work_dir, on_event=print, cancelled=lambda: False)
    results["chair"]      # {"state": "done", "result": {...}} | {"state": "failed", "error": {"kind", "detail"}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor, nn

# The parts of a prop in the order a batch goes through them. Every prop of a wave passes one part before any prop
# starts the next, the multi-view props first.
PARTS = ("cutout", "camera", "features", "ss", "cond_lr", "lr", "upsample", "cond_hr", "hr", "tex", "finish")

# Why a prop failed (PropFailed.kind, and the "kind" of a result's error).
FAILURE_KINDS = ("input", "mv_missing", "oom", "ram", "disk", "missing_file", "error")


@dataclass(frozen=True)
class PropRequest:
    """One prop to make."""
    id: str
    seed: int                       # >= 0; the caller resolves "random"
    resolution: int                 # 1024 | 1536
    multi: bool                     # True: two or more views (the multi-view networks)
    sampler_params: dict            # {"ss": {...}, "shape": {...}, "tex": {...}}: steps, guidance_strength,
    #                                 guidance_rescale, rescale_t; laid over the pipeline file's own
    max_num_tokens: int = 49152
    request_hash: str = ""          # the caller's hash of everything that defines the result; "" = never resumed
    payload: Any = None             # the caller's own object, handed back to the hooks untouched


@dataclass
class Prepared:
    """What hooks.prepare makes of a prop's picture(s)."""
    images: list                    # PIL: single view [RGB, cut out and framed]; multi: framed RGBA views, front first
    meta: dict                      # JSON-able; handed back to hooks.bundle (multi: {"azimuths": [...]})


@dataclass
class DecodedMesh:
    """The decoded prop as hooks.finish gets it. All four tensors are on the CPU: nothing of the prop is left on
    the card when the hook starts."""
    vertices: Tensor                # f32 [Vn,3]
    faces: Tensor                   # i32 [F,3]
    coords: Tensor                  # i32 [N4,3]: the voxels of the texture volume
    attrs: Tensor                   # f32 [N4,6] in 0..1, laid out as attr_layout says
    resolution: int                 # the resolution used (below the request's when a 1536 prop had too many tokens)
    attr_layout: dict               # {"base_color": slice(0, 3), "metallic": ..., "roughness": ..., "alpha": ...}


@dataclass
class Event:
    """One thing that happened in a run. Handed to on_event on the GPU thread."""
    kind: str                       # "part_start" | "part_end" | "prop_start" | "step" | "prop_rest" | "prop_done"
    #                                 | "prop_failed" | "prop_spilled" (the memory cap was lifted for this prop-part)
    #                                 | "prop_resumed" | "net_load" | "net_ready" | "net_drop" | "store_warning"
    part: str = ""
    prop_id: str | None = None
    step: int = 0
    total: int = 0
    detail: dict = field(default_factory=dict)      # prop_done: {"result", "seconds"}; prop_failed: {"kind", "detail"};
    #                                                 net_*: {"name", "seconds", "bytes"}; part_start: {"props": [...]}


class PropFailed(Exception):
    """Raise from a part or a hook to fail ONE prop; the batch goes on."""

    def __init__(self, detail: str, kind: str = "error"):
        super().__init__(detail)
        self.detail = detail
        self.kind = kind            # one of FAILURE_KINDS


class Cancelled(Exception):
    """Raised at a cancel check point. Never recorded as a failure."""


class BatchBroken(Exception):
    """The CUDA context is unusable: the process must be restarted."""

    def __init__(self, detail: str, kind: str = "restart"):
        super().__init__(detail)
        self.detail = detail
        self.kind = kind


class Hooks(Protocol):
    """The caller's side of a prop: everything that touches its files. All called on the GPU thread; each may raise
    PropFailed."""

    def prepare(self, req: PropRequest, matte: Callable[[Image], Image]) -> Prepared:
        """Open, cut out and frame the prop's picture(s). `matte(rgb picture) -> the picture with its alpha set`
        is background removal on the card; its network is loaded the first time it is called."""

    def camera(self, req: PropRequest, image: Image, moge: nn.Module) -> dict:
        """Single view only: {"camera_angle_x", "distance", "mesh_scale"} as Python floats."""

    def bundle(self, req: PropRequest, images: list, meta: dict) -> dict:
        """Multi view only: the views dict of Pixal3DMVImageTo3DPipeline.run_mv - {"images": {512: [1,V,3,512,512],
        1024: [1,V,3,1024,1024]} f32 on the CPU, "camera_angle_x": [1,V], "camera_distance": [1,V],
        "transform_matrix": [1,V,4,4], "mesh_scale": float}."""

    def prop_scope(self, req: PropRequest, part: str) -> ContextManager[dict]:
        """Around the work of one prop in `part` (today only "ss"): yields a dict the hook has filled when the
        block is left; the engine keeps it with the prop's notes."""

    def finish(self, req: PropRequest, mesh: DecodedMesh, notes: dict) -> dict:
        """Write the prop's file from the decoded mesh. Returns what the result carries beside the engine's own
        keys."""


@dataclass
class EngineOptions:
    pipeline_dir: Path              # the snapshot folder: pipeline.json, pipeline_mv.json, ckpts/
    builders: dict[str, Callable[[str], Any]]       # "rembg", "moge", "dino", "naf": (device) -> the module on it;
    #                                                 staged.builders.default_builders(...) outside tests
    device: str = "cuda"
    max_vram_mib: int | None = None                 # tests: behave as on a card of this size
    store_ram_mib: int | None = None                # None: min(768, 5% of RAM), 256 on PCs with 16 GiB or less
    prefetch: str = "cache"                         # "off" | "cache"
    wave_size: int | None = None                    # None / 0: every prop passes each part before the next part
    #                                                 starts; K: K props at a time. run() may override it per batch
    check_rng: bool = False                         # tests: a part that must not draw random numbers and does, fails
