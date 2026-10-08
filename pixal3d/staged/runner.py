"""The order of a batch: every prop of a wave goes through one part before any prop starts the next, the multi-view
props first; a part's networks are on the card only while that part runs.

The runner knows nothing of what a part computes. It is given a context (loader, store, hooks, the caller's event
and cancel callbacks) and a plan - the part functions and what each needs (parts.py is the plan of the engine) -
and it owns everything around them:

- the ladder of a prop-part: normal -> tight -> memory cap lifted. Only an out-of-memory failure goes down it; a
  prop that needed the last step is done and marked slow, one that fails there too is failed;
- failure isolation: a failing prop is closed and the part goes on with the next one;
- cancel: checked before every prop (the parts check inside); everything unfinished is closed;
- resume: a prop whose finished latents a killed run left in the work folder runs only its last part;
- the events the caller's progress is made of.

One rule shapes the code: nothing is retried, closed or trimmed while an exception is being handled. A caught
exception's traceback holds the part function's frames and every tensor in them, so memory freed there is not free.
`_attempt` and `_try_enter` reduce a failure to strings and return; only then the runner acts on it.

Everything here runs on the GPU thread.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .types import PARTS, BatchBroken, Cancelled, Event, PropFailed, PropRequest

MODES = ("normal", "tight", "uncapped")
KINDS = ("mv", "sv")                # the order inside every part: multi-view props first

# MiB outside PyTorch's allocator while a part runs (cuBLAS / cuDNN handles, flash_attn, natten, flex_gemm work
# areas; for "finish" its decode phase, which adds cumesh's fill_holes). The loader's cap leaves this much free.
NATIVE_RESERVE = {**{part: 200 for part in PARTS if part != "finish"}, "finish": 300}
# Both phases of "finish" leave more of the card alone than the other parts do: the page's viewer may load a
# finished model while the next prop is decoded.
FINISH_SAFETY_MIB = 500

# Parts whose networks run cuDNN convolutions. cuDNN quietly takes another algorithm when the work area of its
# first choice cannot be allocated, and another algorithm rounds differently: under the cap the same seed could
# give another cut-out, camera or conditioning without any error. PyTorch counts the failed allocation, so a
# counter that moved during such a part is treated as an out-of-memory failure of that attempt.
CONV_PARTS = frozenset({"cutout", "camera", "features", "cond_lr", "cond_hr", "ss"})
# Parts that draw from the CPU random generator; each starts from the state its predecessor stored.
DRAWING_PARTS = frozenset({"ss", "lr", "hr", "tex"})
# Parts after which PyTorch's cache is given back at once: the conditioning passes and the up-sampling decoder
# leave large freed blocks that the next prop's part does not reuse.
TRIMMED_PARTS = frozenset({"features", "cond_lr", "upsample", "cond_hr"})

# Texts of a failure after which the CUDA context may be gone for good; a probe allocation decides.
BROKEN_CONTEXT = ("cuda error", "device-side assert", "unspecified launch failure")
OOM_TEXTS = ("out of memory", "alloc_failed", "cuda error: 2[", "cudamalloc")
RAM_TEXTS = ("paging file", "defaultcpuallocator")


def glb_reserve(faces: int) -> int:
    """MiB of native memory the caller's finish hook needs for a decoded mesh of `faces` faces (to_glb: the
    nvdiffrast context, one BVH at a time, cumesh)."""
    return 2300 + 60 * faces // 1_000_000


def classify(exc: BaseException) -> str:
    """The kind of a failure, one of types.FAILURE_KINDS. Only "oom" is tried again further down the ladder."""
    if isinstance(exc, PropFailed):
        return exc.kind
    if isinstance(exc, torch.OutOfMemoryError):
        return "oom"
    if isinstance(exc, MemoryError):
        return "ram"
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    text = str(exc).lower()
    if any(words in text for words in RAM_TEXTS):
        return "ram"
    if isinstance(exc, RuntimeError) and any(words in text for words in OOM_TEXTS):
        return "oom"
    return "error"


def describe(exc: BaseException) -> str:
    """The words kept of a failure once its exception is gone."""
    if isinstance(exc, PropFailed):
        return exc.detail
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"FileNotFoundError: {exc.filename}"
    return f"{type(exc).__name__}: {exc}"


def num_ooms() -> int:
    """PyTorch's own count of failed allocations on the card (a host-side counter, no sync)."""
    return int(torch.cuda.memory_stats().get("num_ooms", 0))


def probe(device: str) -> bool:
    """Can the card still be used: one small allocation, written and waited for."""
    try:
        torch.empty(1 << 20, device=device).fill_(1)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        return True
    except Exception:
        return False


@dataclass
class PipelineArgs:
    """What a pipeline file (pipeline.json, pipeline_mv.json) says beside the networks."""
    samplers: dict[str, Any]                # "ss", "shape", "tex" -> the sampler object
    sampler_params: dict[str, dict]         # the same keys -> the file's params; a request's are laid over them
    shape_normalization: dict               # {"mean": [...], "std": [...]}
    tex_normalization: dict
    pbr_attr_layout: dict


@dataclass
class PropState:
    """One prop of a batch as the runner follows it."""
    req: PropRequest
    state: str = "waiting"          # "waiting" | "working" | "resting" | "done" | "failed" | "cancelled"
    part: str = ""                  # the last part begun
    resumed: bool = False           # its latents came from a checkpoint: only "finish" runs
    views: int = 0                  # pictures the cut-out stored (image_0 ..)
    seconds: float = 0.0            # GPU-thread seconds of its own prop-parts
    resolution: int = 0             # the resolution used, known after "finish"
    result: dict | None = None      # what hooks.finish returned
    error: dict | None = None       # {"kind", "detail"}

    @property
    def id(self) -> str:
        return self.req.id

    @property
    def kind(self) -> str:
        return "mv" if self.req.multi else "sv"

    @property
    def over(self) -> bool:
        return self.state in ("done", "failed", "cancelled")


@dataclass
class PartContext:
    """What a part function is given beside its prop."""
    engine: Any                     # the StagedEngine: part "ss" calls engine.sample_sparse_structure through it
    loader: Any                     # NetworkLoader
    store: Any                      # PropStore of this batch
    hooks: Any                      # types.Hooks
    options: Any                    # types.EngineOptions
    emit: Callable[[Event], None]
    cancelled: Callable[[], bool]
    pipeline_args: dict[str, PipelineArgs]          # "sv" / "mv"
    device: str
    inputs: dict = field(default_factory=dict)      # (prop id, key) -> the device tensors taken from the store
    keep: frozenset = frozenset()   # networks the next group needs: `leave` does not drop them
    entering: str = ""              # the network `enter` is loading
    part: str = ""                  # the part that runs

    def take(self, prop: PropState, key: str) -> dict:
        """store.take, remembered until the prop-part ends: a second attempt gets the same device tensors."""
        held = self.inputs.get((prop.id, key))
        if held is None:
            held = self.inputs[(prop.id, key)] = self.store.take(prop.id, key)
        return held

    def args(self, prop: PropState) -> PipelineArgs:
        return self.pipeline_args[prop.kind]

    def check(self) -> None:
        """A cancel check point."""
        if self.cancelled():
            raise Cancelled()


class StagedRunner:
    """Runs one batch. `plan` is parts.py or anything shaped like it: TABLE (part -> fn(ctx, prop, mode)),
    enter(ctx, part, kind), leave(ctx, part, kind), networks(part, kind), applies(part, prop), wants(part),
    produces(part), discards(part, prop), LIGHT (networks not worth reading ahead)."""

    def __init__(self, ctx: PartContext, plan: Any, *, check_rng: bool = False):
        self.ctx = ctx
        self.plan = plan
        self.check_rng = check_rng
        self.props: list[PropState] = []
        self.waves: list[list[PropState]] = []
        self._ahead = ""            # the network last handed to loader.prefetch

    # ---- the batch ------------------------------------------------------------------------------------------------

    def run(self, requests: list[PropRequest], *, wave_size: int | None = None,
            resume: bool = False) -> dict[str, dict]:
        """{prop id: {"state": "done", "result"} | {"state": "failed", "error"} | {"state": "cancelled"}}.
        Raises BatchBroken when the card cannot be used any more; nothing is cleaned up then."""
        ctx = self.ctx
        self._begin(requests, wave_size, resume)
        stopped = False
        try:
            for index in range(len(self.waves)):
                self._wave(index)
        except Cancelled:
            stopped = True
        ctx.inputs.clear()
        ctx.part = ""
        ctx.loader.drop_all()
        if stopped:
            for prop in self.props:
                if not prop.over:
                    prop.state = "cancelled"
                    ctx.store.close_prop(prop.id)
        ctx.loader.prefetch(None)
        if ctx.loader.capped():
            ctx.loader.clear_cap()
        ctx.store.flush()
        return {prop.id: self._report(prop) for prop in self.props}

    def _begin(self, requests: list[PropRequest], wave_size: int | None, resume: bool) -> None:
        ctx, store = self.ctx, self.ctx.store
        ids = [request.id for request in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("two props of a batch have the same id")
        self.props = [PropState(req=request) for request in requests]
        keep = {prop.id: prop.req.request_hash for prop in self.props if prop.req.request_hash} if resume else {}
        found = store.begin(keep)
        for prop in self.props:
            request = prop.req
            store.open_prop(prop.id)
            store.put_small(prop.id, "request", {
                "id": request.id, "seed": request.seed, "resolution": request.resolution, "multi": request.multi,
                "sampler_params": request.sampler_params, "max_num_tokens": request.max_num_tokens,
                "request_hash": request.request_hash})
            if prop.id in found and self._restore(prop, found[prop.id]):
                prop.resumed = True
                ctx.emit(Event("prop_resumed", "finish", prop.id))
            else:
                store.put_small(prop.id, "notes", {"slow": []})
        for prop in self.props:
            if prop.kind == "mv" and not prop.resumed and not ctx.engine.multiview:
                self._fail(prop, "", ("mv_missing", "The multi-view networks are not installed.", ""))

        order = [prop for kind in KINDS for prop in self.props if prop.kind == kind]
        fresh = [prop for prop in order if not prop.resumed and not prop.over]
        size = wave_size if wave_size and wave_size > 0 else max(len(fresh), 1)
        waves = [fresh[start:start + size] for start in range(0, len(fresh), size)] or [[]]
        first = set(map(id, waves[0])) | {id(prop) for prop in order if prop.resumed}
        waves[0] = [prop for prop in order if id(prop) in first]
        self.waves = waves

    def _restore(self, prop: PropState, checkpoint: dict) -> bool:
        """Put a checkpoint's latents where part "finish" reads them. False when it does not hold them all."""
        tensors = checkpoint.get("tensors", checkpoint)
        try:
            coords, res = tensors["coords"], tensors["res"]
            shape_slat, tex_slat = tensors["shape_slat"], tensors["tex_slat"]
        except KeyError:
            return False
        notes = dict(checkpoint.get("notes") or {})
        notes["slow"] = list(notes.get("slow") or [])
        store = self.ctx.store
        store.put_small(prop.id, "coords_hr", {"coords": coords, "res": res})
        store.put_small(prop.id, "shape_slat", {"feats": shape_slat})
        store.put_small(prop.id, "tex_slat", {"feats": tex_slat})
        store.put_small(prop.id, "notes", notes)
        return True

    def _report(self, prop: PropState) -> dict:
        if prop.state == "done":
            return {"state": "done", "result": prop.result}
        if prop.state == "failed":
            return {"state": "failed", "error": prop.error}
        return {"state": "cancelled"}

    # ---- waves, parts, groups -------------------------------------------------------------------------------------

    def _alive(self, part: str, wave: list[PropState], kind: str | None = None) -> list[PropState]:
        """The props of a wave that still go through `part`, in run order."""
        return [prop for prop in wave
                if not prop.over and self.plan.applies(part, prop) and (part == "finish" or not prop.resumed)
                and (kind is None or prop.kind == kind)]

    def _wave(self, index: int) -> None:
        ctx, wave = self.ctx, self.waves[index]
        for part in PARTS:
            alive = self._alive(part, wave)
            if not alive:
                continue
            ctx.part = part
            ctx.emit(Event("part_start", part, detail={"props": [prop.id for prop in alive], "wave": index + 1,
                                                       "waves": len(self.waves)}))
            done = 0
            for kind in KINDS:
                group = [prop for prop in alive if prop.kind == kind]
                if group:
                    self._group(index, part, kind, group, done, len(alive))
                    done += len(group)
            ctx.store.flush()
            ctx.emit(Event("part_end", part))

    def _later(self, index: int, part: str, kind: str) -> list[tuple[str, ...]]:
        """The networks of every group that runs after (part, kind) of wave `index`, group by group: the rest of
        the wave, then the next wave."""
        groups = []
        here = (PARTS.index(part), KINDS.index(kind))
        for wave_index in (index, index + 1):
            if wave_index >= len(self.waves):
                break
            for part_index, later_part in enumerate(PARTS):
                for kind_index, later_kind in enumerate(KINDS):
                    if wave_index == index and (part_index, kind_index) <= here:
                        continue
                    if self._alive(later_part, self.waves[wave_index], later_kind):
                        groups.append(tuple(self.plan.networks(later_part, later_kind)))
        return groups

    def _group(self, index: int, part: str, kind: str, group: list[PropState], done: int, count: int) -> None:
        ctx, store = self.ctx, self.ctx.store
        failure = self._enter(part, kind)
        if failure is not None:
            if kind == "mv" and failure[0] == "missing_file" and ctx.entering.endswith(".mv"):
                failure = ("mv_missing", failure[1], failure[2])
            for prop in group:
                self._fail(prop, part, failure)
            return
        self._read_ahead(index, part, kind)
        self._want(part, group[0])
        for position, prop in enumerate(group):
            ctx.check()
            if position + 1 < len(group):
                self._want(part, group[position + 1])
            store.pump()
            self._prop_part(part, prop, done + position + 1, count)
        later = self._later(index, part, kind)
        ctx.keep = frozenset(later[0]) if later else frozenset()
        self.plan.leave(ctx, part, kind)

    def _read_ahead(self, index: int, part: str, kind: str) -> None:
        """While a group runs, the file of the next network that will be loaded is read into the system's file
        cache: the first one a later group needs that is neither on the card nor this group's own."""
        loader = self.ctx.loader
        skip = set(self.plan.networks(part, kind)) | set(loader.loaded()) | set(self.plan.LIGHT)
        ahead = [name for names in self._later(index, part, kind) for name in names if name not in skip]
        if ahead and ahead[0] != self._ahead:
            self._ahead = ahead[0]
            loader.prefetch(ahead[0])

    def _want(self, part: str, prop: PropState) -> None:
        for key in self.plan.wants(part):
            self.ctx.store.want(prop.id, key)

    def _enter(self, part: str, kind: str) -> tuple[str, str, str] | None:
        """Load the group's networks; after a failure everything is dropped and it is tried once more."""
        loader = self.ctx.loader
        failure = None
        for _ in range(2):
            failure = self._try_enter(part, kind)
            if failure is None:
                return None
            loader.drop_all()
            loader.trim()
        return failure

    def _try_enter(self, part: str, kind: str) -> tuple[str, str, str] | None:
        try:
            self.plan.enter(self.ctx, part, kind)
        except (Cancelled, BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            return classify(exc), describe(exc), traceback.format_exc()
        return None

    # ---- one prop in one part -------------------------------------------------------------------------------------

    def _attempt(self, part: str, prop: PropState, mode: str) -> tuple[str, str, str] | None:
        """One try of a prop-part. A failure comes back as (kind, words, traceback text): when this returns, the
        exception, its traceback and the part function's frames are gone."""
        try:
            self.plan.TABLE[part](self.ctx, prop, mode)
        except (Cancelled, BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            return classify(exc), describe(exc), traceback.format_exc()
        return None

    def _prop_part(self, part: str, prop: PropState, position: int, count: int) -> None:
        ctx, loader, store = self.ctx, self.ctx.loader, self.ctx.store
        prop.state, prop.part = "working", part
        ctx.emit(Event("prop_start", part, prop.id, detail={"position": position, "of": count}))
        started = time.perf_counter()
        failure = None
        try:
            for mode in MODES:
                if mode == "uncapped":
                    loader.clear_cap()
                    self._note_slow(prop, part)
                    ctx.emit(Event("prop_spilled", part, prop.id))
                elif part != "finish":                  # "finish" sets its own cap, once per phase
                    loader.set_cap(NATIVE_RESERVE[part])
                ooms = num_ooms()
                rng = torch.get_rng_state() if self.check_rng and part not in DRAWING_PARTS else None
                failure = self._attempt(part, prop, mode)
                if failure is None and part in CONV_PARTS and num_ooms() > ooms:
                    failure = ("oom", "an allocation failed inside a convolution", "")
                    self._discard(prop, self.plan.produces(part))       # nobody has read them yet
                if failure is None and rng is not None and not torch.equal(torch.get_rng_state(), rng):
                    failure = ("error", f"part {part} drew from the CPU random generator, which it must not", "")
                if failure is None:
                    break
                loader.trim()
                if failure[0] == "error" and any(words in failure[1].lower() for words in BROKEN_CONTEXT) \
                        and not probe(ctx.device):
                    raise BatchBroken(failure[1])
                if failure[0] != "oom" or mode == "uncapped":
                    break
        finally:
            ctx.inputs.clear()
            prop.seconds += time.perf_counter() - started
        if failure is not None:
            self._fail(prop, part, failure)
            return
        self._discard(prop, self.plan.discards(part, prop))
        if part in TRIMMED_PARTS:
            loader.trim()
        if part != "finish":
            prop.state = "resting"
            ctx.emit(Event("prop_rest", part, prop.id))
            return
        notes = store.get_small(prop.id, "notes")
        prop.result = {**(prop.result or {}), "slow": list(notes.get("slow") or []), "seconds": prop.seconds,
                       "resolution": prop.resolution, "resumed": prop.resumed}
        store.close_prop(prop.id)
        prop.state = "done"
        ctx.emit(Event("prop_done", part, prop.id, detail={"result": prop.result, "seconds": prop.seconds}))

    def _discard(self, prop: PropState, keys: tuple[str, ...]) -> None:
        if keys:
            self.ctx.store.discard(prop.id, *keys)

    def _note_slow(self, prop: PropState, part: str) -> None:
        store = self.ctx.store
        notes = dict(store.get_small(prop.id, "notes"))
        notes["slow"] = [*(name for name in notes.get("slow") or [] if name != part), part]
        store.put_small(prop.id, "notes", notes)

    def _fail(self, prop: PropState, part: str, failure: tuple[str, str, str]) -> None:
        kind, detail, trace = failure
        print(f"[staged] prop {prop.id} failed{' in ' + part if part else ''} ({kind}): {detail}", flush=True)
        if trace:
            print(trace, flush=True)
        self.ctx.store.close_prop(prop.id)
        self.ctx.loader.trim()
        prop.state = "failed"
        prop.error = {"kind": kind, "detail": detail}
        self.ctx.emit(Event("prop_failed", part, prop.id, detail=dict(prop.error)))
