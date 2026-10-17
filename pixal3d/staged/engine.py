"""The staged engine: Pixal3D's cascade run part by part over a whole batch of props, with only the networks of the
running part on the card.

    engine = StagedEngine(EngineOptions(pipeline_dir=snapshot, builders=default_builders(default_paths(models))),
                          hooks)                 # reads the two pipeline files; loads nothing, touches no card
    results = engine.run(requests, work_dir, on_event=record, cancelled=stop.is_set)     # on the GPU thread
    engine.release()                             # before anything else needs the card (the reference pipeline)
    engine.close()                               # at exit

The engine is the glue: a NetworkLoader that knows where every network lies (loader.py), a PropStore per batch in
the batch's work folder (store.py), the parts (parts.py) and the runner that orders them (runner.py). The caller's
hooks do everything that touches a prop's files (types.Hooks). Results are those of the reference pipelines
(Pixal3DImageTo3DPipeline.run, Pixal3DMVImageTo3DPipeline.run_mv) for the same picture, seed and settings.

`engine.sample_sparse_structure` is the one attribute a caller may wrap (a guard that looks at the sparse
structure and draws it again): part `ss` calls it through the engine, positionally, as the reference's run() calls
the pipeline's method of that name.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from ..pipelines import samplers
from . import parts
from ..modules import int8, sage
from .runner import PartContext, PipelineArgs, StagedRunner
from .types import EngineOptions, Event, Hooks, PropRequest

# The engine's network names and the keys of a pipeline file's "models". The four flow networks exist once per
# kind ("ss_flow.sv" from pipeline.json, "ss_flow.mv" from pipeline_mv.json); the decoders serve both kinds.
FLOWS = {"ss_flow": "sparse_structure_flow_model", "lr_flow": "shape_slat_flow_model_512",
         "hr_flow": "shape_slat_flow_model_1024", "tex_flow": "tex_slat_flow_model_1024"}
DECODERS = {"ss_dec": "sparse_structure_decoder", "shape_dec": "shape_slat_decoder", "tex_dec": "tex_slat_decoder"}
SAMPLERS = {"ss": "sparse_structure_sampler", "shape": "shape_slat_sampler", "tex": "tex_slat_sampler"}
BUILDERS = ("rembg", "moge", "dino", "naf")
# Libraries the flows' first pass of a process imports when it first needs them, on the GPU thread inside the first
# sparse-structure pass (SageAttention's import alone: 0.5-0.8 s); none of them touches the card when imported.
KERNEL_LIBRARIES = ("triton", "flash_attn")


def _prepare_flow(model: Any, kind: str = "") -> Any:
    """A flow network (`kind`: ss, lr, hr or tex) once its weights are on the card (the loader's convert hook):
    SageAttention's scaling of v folded into its weights when the switch is on (modules/sage.py, exact), then its
    token-sized linears as int8 GEMMs where PIXAL3D_INT8's policy says so for that kind (modules/int8.py; the
    fold comes first so that the int8 scales carry it)."""
    try:
        from ..modules.sparse import config as sparse_config

        if sparse_config.SAGE:
            sage.fold(model)
    except Exception as exc:        # noqa: BLE001 - a network that cannot be folded runs as before
        print(f"[SAGE] the fold of {type(model).__name__} failed ({type(exc).__name__}: {str(exc)[:200]}); the "
              f"network runs unfolded", flush=True)
    return int8.convert(model, kind)


def _import_ahead() -> None:
    """Import the flows' kernel libraries (and SageAttention's, when the switch is on), on a thread of its own: by
    the time a batch reaches its first flow they are there. A library that cannot be imported is left to the first
    pass, which handles it as it did."""
    names = list(KERNEL_LIBRARIES)
    try:
        from ..modules.attention import config

        if config.SAGE:
            names.append("sageattention")
    except Exception:       # noqa: BLE001 - the first pass imports what it needs as before
        pass
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:   # noqa: BLE001
            pass


def _pipeline_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["args"]


def _checkpoint_present(checkpoint: Path) -> bool:
    return Path(f"{checkpoint}.json").is_file() and Path(f"{checkpoint}.safetensors").is_file()


def _pipeline_args(config: dict) -> PipelineArgs:
    """The samplers and numbers of a pipeline file, as Pixal3DImageTo3DPipeline.from_pretrained reads them."""
    return PipelineArgs(
        samplers={stage: getattr(samplers, config[key]["name"])(**config[key]["args"])
                  for stage, key in SAMPLERS.items()},
        sampler_params={stage: config[key]["params"] for stage, key in SAMPLERS.items()},
        shape_normalization=config["shape_slat_normalization"],
        tex_normalization=config["tex_slat_normalization"],
        pbr_attr_layout={"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5),
                         "alpha": slice(5, 6)},
    )


def _builder_file(name: str, build: Callable[[str], Any]) -> Path | None:
    """The weights file one of staged.builders' builders reads (it carries its ModelPaths), so that the loader
    can read it ahead; None for any other builder."""
    paths = next((arg for arg in getattr(build, "args", ()) if hasattr(arg, "naf_weights")), None)
    if paths is None:
        return None
    file = {"rembg": Path(paths.rembg) / "model.safetensors", "moge": Path(paths.moge),
            "dino": Path(paths.dino) / "model.safetensors", "naf": Path(paths.naf_weights)}[name]
    return file if file.is_file() else None


class StagedEngine:
    def __init__(self, options: EngineOptions, hooks: Hooks, *, loader_factory: Callable[..., Any] | None = None,
                 store_factory: Callable[..., Any] | None = None):
        """Reads the pipeline files, tells the loader where the networks lie, builds the samplers and starts
        reading the first network's file ahead into the system's file cache. No weights are loaded and CUDA is not
        initialised. The two factories replace NetworkLoader and PropStore (tests)."""
        self.options = options
        self.hooks = hooks
        self.device = options.device
        folder = Path(options.pipeline_dir)
        files = {"sv": _pipeline_file(folder / "pipeline.json")}
        if (folder / "pipeline_mv.json").is_file():
            multi = _pipeline_file(folder / "pipeline_mv.json")
            if all(_checkpoint_present(folder / multi["models"][key]) for key in FLOWS.values()):
                files["mv"] = multi
        self.multiview = "mv" in files      # pipeline_mv.json and its four checkpoints are on disk
        self.pipeline_args = {kind: _pipeline_args(config) for kind, config in files.items()}

        if loader_factory is None:
            from .loader import NetworkLoader as loader_factory
        self._store_factory = store_factory
        self.loader = loader_factory(options.device, prefetch=options.prefetch, max_vram_mib=options.max_vram_mib,
                                     on_event=self._relay)
        for kind, config in files.items():
            for name, key in FLOWS.items():
                self.loader.register_checkpoint(f"{name}.{kind}", folder / config["models"][key],
                                                convert=functools.partial(_prepare_flow, kind=name.split("_")[0]))
        for name, key in DECODERS.items():
            self.loader.register_checkpoint(name, folder / files["sv"]["models"][key])
        for name in BUILDERS:
            build = options.builders[name]
            self.loader.register_builder(name, build, _builder_file(name, build))
        # The first network a batch loads (background removal, at its first cut-out: 0.9 GB, read from a cold disk
        # in the first batch otherwise) is read into the system's file cache now, by the loader's read-ahead
        # thread; nothing is kept in the process and the card is not touched. The runner does the same after a
        # batch's last network, for the next batch.
        self.loader.prefetch(parts.FIRST_NETWORK)
        if torch.device(options.device).type == "cuda":
            threading.Thread(target=_import_ahead, name="p3d-import-ahead", daemon=True).start()

        self._store: Any = None             # the store, context and event callback of the batch that runs
        self._ctx: PartContext | None = None
        self._on_event: Callable[[Event], None] | None = None
        self._draw: Callable[..., torch.Tensor] | None = None

    def run(self, props: list[PropRequest], work_dir: Path, *, on_event: Callable[[Event], None],
            cancelled: Callable[[], bool], wave_size: int | None = None, resume: bool = False) -> dict[str, dict]:
        """Make the props. Call on the GPU thread; blocks until every prop is done, failed or cancelled.

        -> {prop id: {"state": "done", "result": {...}} | {"state": "failed", "error": {"kind", "detail"}}
                     | {"state": "cancelled"}}. A result carries what hooks.finish returned and "slow" (the parts
        that needed the memory cap lifted; empty for a prop that stayed inside it), "seconds", "resolution",
        "resumed". Raises BatchBroken when the card cannot be used any more.

        work_dir is emptied first; with resume=True the latents a killed run left there for these props
        (same id and request_hash) are kept, and those props are only decoded and saved.
        wave_size: props that go through the parts together (None: the options' value; 0: all of them)."""
        if self._store is not None:
            raise RuntimeError("the staged engine runs one batch at a time")
        store_factory = self._store_factory
        if store_factory is None:
            from .store import PropStore as store_factory
        # None for a size: the store's own choice for this PC (smaller on one with 16 GiB of RAM or less)
        store = store_factory(Path(work_dir), self.device, ram_mib=self.options.store_ram_mib, slots=2,
                              slot_mib=None, on_event=self._relay)
        ctx = PartContext(engine=self, loader=self.loader, store=store, hooks=self.hooks, options=self.options,
                          emit=on_event, cancelled=cancelled, pipeline_args=self.pipeline_args, device=self.device)
        self._store, self._ctx, self._on_event = store, ctx, on_event
        runner = StagedRunner(ctx, parts, check_rng=self.options.check_rng)
        try:
            with torch.no_grad():
                results = runner.run(list(props), resume=resume,
                                     wave_size=self.options.wave_size if wave_size is None else wave_size)
        except BaseException:
            with contextlib.suppress(Exception):    # the card may be unusable: the first failure is the one to tell
                store.close()
            raise
        finally:
            self._store = self._ctx = self._on_event = self._draw = None
        store.close()
        return results

    def sample_sparse_structure(self, cond: dict, resolution: int, num_samples: int = 1,
                                sampler_params: dict | None = None) -> torch.Tensor:
        """One draw of the sparse structure of the prop part `ss` is working on: coords [N0,4] int32 on the card.
        `cond` is {"cond": {"global", "proj"}, "neg_cond": {...}}; `sampler_params` are laid over the pipeline
        file's. A wrapper put on the engine under this name is what `ss` calls."""
        if self._draw is None:
            raise RuntimeError("sample_sparse_structure can only be called while part 'ss' works on a prop")
        return self._draw(cond, resolution, num_samples, sampler_params)

    @contextlib.contextmanager
    def drawing(self, draw: Callable[..., torch.Tensor]) -> Iterator[None]:
        """Part `ss` around its call of sample_sparse_structure: `draw` makes the draws for its prop."""
        self._draw = draw
        try:
            yield
        finally:
            self._draw = None

    def _relay(self, event: dict) -> None:
        """An event of the loader or the store ({"kind": "net_load", "name", ...}), handed on to the caller."""
        if self._on_event is None:
            return
        detail = {name: value for name, value in event.items() if name not in ("kind", "prop_id")}
        self._on_event(Event(str(event.get("kind", "")), self._ctx.part if self._ctx is not None else "",
                             event.get("prop_id"), detail=detail))

    def release(self) -> None:
        """Drop every network and lift the memory cap: the card is as if the engine had never run."""
        self.loader.prefetch(None)
        self.loader.drop_all()
        if self.loader.capped():
            self.loader.clear_cap()
        proj_cond = sys.modules.get("pixal3d.pipelines.proj_cond")      # only the parts import it, and lazily
        if proj_cond is not None:
            proj_cond.clear_grids()
        if torch.cuda.is_initialized():
            # cuBLAS keeps an 8 MiB work area from a process's first matrix product on; the next tool on the card
            # can use it, and the next batch makes it again at once.
            torch._C._cuda_clearCublasWorkspaces()

    def close(self) -> None:
        """release(), then stop the loader's thread. The store's belongs to a batch and is stopped with it."""
        self.release()
        self.loader.close()
