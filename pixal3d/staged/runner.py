"""The order of a batch: every prop of a wave goes through one part before any prop starts the next, the multi-view
props first; a part's networks are on the card only while that part runs.

The runner knows nothing of what a part computes. It is given a context (loader, store, hooks, the caller's event
and cancel callbacks) and a plan - the part functions and what each needs (parts.py is the plan of the engine) -
and it owns everything around them:

- the ladder of a prop-part: normal -> tight -> (when the options allow a spill) memory cap lifted. Only an
  out-of-memory failure goes down it; a prop that needed the last step is done and marked slow, one that fails
  there too is failed;
- failure isolation: a failing prop is closed and the part goes on with the next one;
- cancel: checked before every prop (the parts check inside); everything unfinished is closed;
- resume: a prop whose finished latents a killed run left in the work folder runs only its last part;
- the finish steps a hook hands back (types.FinishSteps): their GPU steps run here between the other props'
  turns, their CPU steps on WORKERS worker threads meanwhile; before each step the hook is told how much it may keep
  on the card for its later steps (plan.tail_room: what the props still to be decoded leave), and what it holds
  is let go when a decode runs out of memory;
- the events the caller's progress is made of.

One rule shapes the code: nothing is retried, closed or trimmed while an exception is being handled. A caught
exception's traceback holds the part function's frames and every tensor in them, so memory freed there is not free.
`_attempt` and `_try_enter` reduce a failure to strings and return; only then the runner acts on it.

Everything here runs on the GPU thread, except the CPU steps of a FinishSteps.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .types import PARTS, BatchBroken, Cancelled, Event, FinishSteps, PropFailed, PropRequest

MODES = ("normal", "tight", "uncapped")
KINDS = ("mv", "sv")                # the order inside every part: multi-view props first
IN_FLIGHT = 3                       # props that may be in their finish steps while another prop is decoded: each
#                                     holds its decoded mesh and its textures in RAM meanwhile (and the tree of its
#                                     BVH, when the engine gave it room: cuBVH keeps a copy in RAM). A prop in its
#                                     last step (the hub's export) no longer counts (_in_flight); a fresh prop's
#                                     first CPU step (the hub's BVH build) is handed over after the boundary's GPU
#                                     steps while props remain to decode (_fresh_tails), so two trees are in RAM
#                                     at that moment rather than three; the pair's props go biggest first
#                                     (plan.pair_order), so the last tail, which nothing hides, is the smallest
WORKERS = 2                         # threads for the CPU steps: one prop's export no longer waits behind another's
#                                     BVH build or unwrap (measured on a tight card: exports of 3 s queued for 23 s
#                                     behind one thread). The steps' libraries (cuBVH's build, xatlas, OpenCV) let
#                                     go of Python's lock, so two run at once; what they hold in RAM is bounded by
#                                     IN_FLIGHT, which counts the props, not the threads
STEP_WAIT = 0.05                    # seconds the GPU thread waits for a CPU step before it looks again

# Texts of a failure after which the CUDA context may be gone for good; a probe allocation decides.
BROKEN_CONTEXT = ("cuda error", "device-side assert", "unspecified launch failure")
OOM_TEXTS = ("out of memory", "alloc_failed", "cuda error: 2[", "cudamalloc")
RAM_TEXTS = ("paging file", "defaultcpuallocator")


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


def num_ooms(device: str) -> int:
    """PyTorch's own count of failed allocations on `device` (a host-side counter, no sync); 0 off the card."""
    if torch.device(device).type != "cuda":
        return 0
    return int(torch.cuda.memory_stats(device).get("num_ooms", 0))


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
    bundled: bool = False           # its multi-view bundle is kept in the store (parts._keep_bundle)
    slow: list = field(default_factory=list)        # parts that needed the memory cap lifted
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

    def noted(self, notes: dict) -> dict:
        """The prop's notes as the hook and the checkpoint get them: with the parts that ran with the cap lifted."""
        return {**notes, "slow": list(self.slow)}


@dataclass
class _Tail:
    """A prop in the steps its finish hook handed back."""
    prop: PropState
    steps: FinishSteps
    index: int = 0                  # the next step to run
    future: Any = None              # the CPU step under way on the worker thread
    offered: int = 0                # the room written before the step under way
    open: bool = False              # a CPU step is under way: its hold counts from the offer on (steps.holds_mib)
    held: bool = False              # a step has held something (state["card_mib"] seen): later CPU steps that say
    #                                 nothing hold nothing


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
    hold: frozenset = frozenset()   # networks the prop-part that runs leaves on the card for the next prop (the
    #                                 paired finish: both decoders, where plan.pair_keep found room for them)
    entering: str = ""              # the network `enter` is loading
    part: str = ""                  # the part that runs

    def take(self, prop: PropState, key: str) -> dict:
        """store.take, remembered until the prop-part ends: a second attempt gets the same device tensors."""
        held = self.inputs.get((prop.id, key))
        if held is None:
            held = self.inputs[(prop.id, key)] = self.store.take(prop.id, key)
        return held

    def args(self, prop: PropState) -> PipelineArgs:
        """The settings of the prop's pipeline file. A resumed multi-view prop runs only "finish", which needs
        nothing a multi-view file says, and the files may be gone by then: it falls back to the single-view one."""
        return self.pipeline_args.get(prop.kind) or self.pipeline_args["sv"]

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
        self._returned: Any = None  # what the last part function returned
        self._tails: list[_Tail] = []
        self._worker: concurrent.futures.ThreadPoolExecutor | None = None
        self._wave_now: list[PropState] = []
        self._extra: set[str] = set()   # networks on the card that the running prop-part does not use, kept there
        #                                 for a later prop because plan.pair_keep found room (_set_aside); dropped
        #                                 at any failure, so that a retry finds the card as it would without them

    def _modes(self) -> tuple[str, ...]:
        """The ladder: the cap is lifted only where the options allow a spill into shared memory."""
        return MODES if getattr(self.ctx.options, "spill", False) else MODES[:2]

    # ---- the batch ------------------------------------------------------------------------------------------------

    def run(self, requests: list[PropRequest], *, wave_size: int | None = None,
            resume: bool = False) -> dict[str, dict]:
        """{prop id: {"state": "done", "result"} | {"state": "failed", "error"} | {"state": "cancelled"}}.
        Raises BatchBroken when the card cannot be used any more; nothing is cleaned up then."""
        ctx = self.ctx
        self._begin(requests, wave_size, resume)
        stopped, failure = False, None
        try:
            for index in range(len(self.waves)):
                self._wave(index)
        except Cancelled:
            stopped = True
        except (BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            failure = (classify(exc), describe(exc), traceback.format_exc())
        # The props that reached their finish steps are finished, cancel or not: their files are being written.
        try:
            self._pump_tails(until=lambda: not self._tails)
        except (BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            failure = failure or (classify(exc), describe(exc), traceback.format_exc())
        failure = self._settle() or failure
        for prop in self.props:
            if prop.over:
                continue
            if failure is not None:
                self._fail(prop, prop.part, failure)
            elif stopped:
                prop.state = "cancelled"
                ctx.store.close_prop(prop.id)
        return {prop.id: self._report(prop) for prop in self.props}

    def _settle(self) -> tuple[str, str, str] | None:
        """Give the card and the work folder back between batches.

        The work between two prop-parts - asking the store for the next inputs, sending them, dropping a group's
        networks, emptying the folder - belongs to no prop, and on a PC that has run out of room it is where a
        failure surfaces. It comes back as a failure of the props that had not finished, so the ones that did
        keep their results and the caller still gets its report.
        """
        ctx = self.ctx
        try:
            if self._worker is not None:
                self._worker.shutdown(wait=True, cancel_futures=True)
                self._worker = None
            for tail in self._tails:                # only after a failure of the steps' machinery itself
                self._fail(tail.prop, "finish", ("error", "the finish steps were not completed", ""))
            self._tails = []
            ctx.inputs.clear()
            ctx.part = ""
            ctx.hold = frozenset()
            self._extra = set()
            ctx.loader.drop_all()
            if self._ahead != getattr(self.plan, "FIRST_NETWORK", None):
                ctx.loader.prefetch(None)       # a read of the next batch's first network goes on (_read_next_batch)
            if ctx.loader.capped():
                ctx.loader.clear_cap()
            ctx.store.flush()
        except (BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            return classify(exc), describe(exc), traceback.format_exc()
        return None

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
                store.put_small(prop.id, "notes", {})
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
        try:
            tensors = checkpoint["tensors"]
            coords, res = tensors["coords"], tensors["res"]
            shape_slat, tex_slat = tensors["shape_slat"], tensors["tex_slat"]
        except KeyError:
            return False
        notes = dict(checkpoint.get("notes") or {})
        prop.slow = list(notes.get("slow") or [])
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

    def _paired(self) -> tuple[str, str] | None:
        """The two last parts when each prop goes through them back to back (EngineOptions.paired_finish): the
        texture and the finish. None: every part for all the props, in turn."""
        if not getattr(self.ctx.options, "paired_finish", True) or PARTS[-2:] != ("tex", "finish"):
            return None
        return "tex", "finish"

    def _wave(self, index: int) -> None:
        ctx, wave = self.ctx, self.waves[index]
        self._wave_now = wave
        paired = self._paired()
        for part in PARTS:
            if paired and part == paired[1]:
                continue                            # ran with its mate
            if paired and part == paired[0]:
                if self._alive(part, wave) or self._alive(paired[1], wave):
                    self._pair(index, *paired)
                continue
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
            if part == PARTS[-1]:
                self._read_next_batch(index)
            if self._tails:
                self._pump_tails(until=lambda: not self._tails)
            ctx.store.flush()
            ctx.emit(Event("part_end", part))

    def _pair(self, index: int, first: str, last: str) -> None:
        """The parts `first` and `last` (the texture and the finish) for the wave, each prop through both back
        to back: a prop is decoded and handed to its finish steps as soon as its texture is sampled, instead of
        after every prop's texture. Its steps' CPU work (the BVH, the UV unwrap, the file) then runs on the worker
        threads while the next prop's texture is sampled - a minute and more, against the seconds of the next
        prop's decode it overlapped with before - and only the last prop's steps are left with nothing to hide
        under. The card pays the texture network's turn-round: it leaves the card before each decode (the decode
        needs the room, and the cap `finish` sets takes what is allocated for room) and is read again for the next
        prop, from the file cache in most cases. What the store sent ahead for the next prop's texture goes back
        to its slot meanwhile, so the decode has the whole card. The events are those of the two parts, their
        brackets overlapping: `first` opens before its first prop, `last` before its first prop, `first` closes
        after its last prop, `last` after the last prop's steps are over. Props that run `last` alone (resumed
        from a checkpoint) come before the pairs of their kind: they are nearly done, and their steps' CPU work
        hides under the textures that follow. A process that dies in a prop's finish leaves the checkpoint of
        that prop alone: the props after it have not sampled their texture yet."""
        ctx, store, wave = self.ctx, self.ctx.store, self._wave_now
        firsts, lasts = self._alive(first, wave), self._alive(last, wave)
        waves = {"wave": index + 1, "waves": len(self.waves)}
        ctx.emit(Event("part_start", first, detail={"props": [prop.id for prop in firsts], **waves}))
        opened = False
        done_first = done_last = 0

        def run_last(prop: PropState) -> None:
            nonlocal opened, done_last
            if not opened:
                opened = True
                ctx.emit(Event("part_start", last, detail={"props": [prop.id for prop in lasts], **waves}))
            done_last += 1
            ctx.part = last
            self._prop_part(last, prop, done_last, len(lasts))

        for kind in KINDS:
            group = self._pair_order([prop for prop in firsts if prop.kind == kind])
            alone = [prop for prop in lasts if prop.kind == kind and prop not in group]
            if not group and not alone:
                continue
            for prop in alone:
                ctx.check()
                store.pump()
                if self._tails:
                    self._pump_tails(until=lambda: self._in_flight() < IN_FLIGHT)
                run_last(prop)
            if not group:
                continue
            failure = self._enter(first, kind)
            if failure is not None:
                if kind == "mv" and failure[0] == "missing_file" and ctx.entering.endswith(".mv"):
                    failure = ("mv_missing", failure[1], failure[2])
                for prop in group:
                    self._fail(prop, first, failure)
                continue
            self._read_ahead(index, first, kind)
            self._want(first, group[0])
            own = set(self.plan.networks(first, kind))
            for position, prop in enumerate(group):
                ctx.check()
                if position + 1 < len(group):
                    self._want(first, group[position + 1])
                store.pump()
                if self._tails:
                    self._pump_tails(until=lambda: self._in_flight() < IN_FLIGHT)
                self._extra -= own                  # a texture network kept beside the last finish is its own again
                ctx.part = first
                self._prop_part(first, prop, done_first + position + 1, len(firsts))
                if prop.over:
                    continue
                following = next((other for other in group[position + 1:] if not other.over), None)
                self._set_aside(first, kind, more=position + 1 < len(group), prop=prop, following=following)
                if not position + 1 < len(group):
                    self._read_ahead(index, first, kind)
                run_last(prop)
                kept = [name for name in ctx.hold if ctx.loader.get(name) is not None]
                ctx.hold = frozenset()
                self._extra |= set(kept)            # the decoders, for the next prop's finish
            done_first += len(group)
            later = self._later(index, last, kind)
            ctx.keep = frozenset(later[0]) if later else frozenset()
            self.plan.leave(ctx, first, kind)
            self.plan.leave(ctx, last, kind)
            self._extra = {name for name in self._extra if ctx.loader.get(name) is not None}
        self._read_next_batch(index)
        store.flush()
        ctx.emit(Event("part_end", first))
        if not opened:                              # no prop reached its finish: the bracket still closes below
            opened = True
            ctx.emit(Event("part_start", last, detail={"props": [prop.id for prop in lasts], **waves}))
        if self._tails:
            self._pump_tails(until=lambda: not self._tails)
        store.flush()
        ctx.emit(Event("part_end", last))

    def _pair_order(self, group: list[PropState]) -> list[PropState]:
        """The order the props of a group go through the pair: the plan's (`plan.pair_order`: the biggest shape
        first, so that the steps left with nothing to hide under - the last prop's - are the smallest prop's),
        or the batch's own where the plan has none. The numbers of a prop do not depend on its place: every
        part draws from the generator state its own prop stored."""
        order = getattr(self.plan, "pair_order", None)
        if order is None or len(group) < 2:
            return group
        try:
            ordered = list(order(self.ctx, group))
        except Exception:
            return group
        return ordered if sorted(map(id, ordered)) == sorted(map(id, group)) else group

    def _set_aside(self, part: str, kind: str, more: bool, prop: PropState | None = None,
                   following: PropState | None = None) -> None:
        """Before a prop's finish in a pair: the part's networks leave the card (with `more` props of the group to
        come, its file is read ahead again meanwhile), and what the store uploaded ahead for the next prop goes
        back to its slot, so the decode has the whole card - unless the plan finds room for them beside everything
        the finish and the next prop's texture need (plan.pair_keep, with `following` the prop that comes next):
        then the texture network stays (it is not read again for `following`), and when there is room for that as
        well, so do both decoders after the finish (ctx.hold: not loaded again for `following`'s finish)."""
        ctx, loader = self.ctx, self.ctx.loader
        own = self.plan.networks(part, kind)
        keep = self._keeps(prop, following) if more and prop is not None and following is not None else frozenset()
        held = [name for name in own if loader.get(name) is not None and name not in keep]
        if held:
            loader.drop(*held)
            if more and held[0] not in self.plan.LIGHT:
                self._ahead = held[0]
                loader.prefetch(held[0])
        ctx.store.pump(upload=False)
        self._extra = {name for name in keep if name in own and loader.get(name) is not None}
        ctx.hold = frozenset(name for name in keep if name not in own)

    def _keeps(self, prop: PropState, following: PropState) -> frozenset:
        """What may stay on the card through `prop`'s finish for `following` (plan.pair_keep): nothing where the
        plan has no such rule or cannot tell."""
        rule = getattr(self.plan, "pair_keep", None)
        if rule is None:
            return frozenset()
        natives = [int(tail.steps.native_mib or 0) for tail in self._tails]
        try:
            return frozenset(rule(self.ctx, prop, following, natives))
        except Exception:
            return frozenset()

    def _drop_extra(self) -> None:
        """The networks kept for a later prop leave the card (a failure: the retry, or the next prop, finds the card
        as it would without them), and the running part keeps nothing for the next prop."""
        loader = self.ctx.loader
        names = [name for name in self._extra if loader.get(name) is not None]
        self._extra = set()
        self.ctx.hold = frozenset()
        if names:
            loader.drop(*names)

    def _extra_mib(self) -> int:
        """MiB the networks kept for a later prop take on the card (those a part leaves for the next prop too)."""
        size = getattr(self.ctx.loader, "size_mib", None)
        if size is None:
            return 0
        return sum(int(size(name) or 0) for name in {*self._extra, *self.ctx.hold})

    def _read_next_batch(self, index: int) -> None:
        """The batch's last network has been read: the first one a batch loads (plan.FIRST_NETWORK, background
        removal) is read into the system's file cache for the next batch, while the last props' steps run."""
        first = getattr(self.plan, "FIRST_NETWORK", None)
        if first and index == len(self.waves) - 1 and self._ahead != first and self.ctx.loader.get(first) is None:
            self._ahead = first
            self.ctx.loader.prefetch(first)

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
            if self._tails:
                # Steps of the props finished before: their GPU steps now, while this prop's inputs are still on
                # their way, and no more than IN_FLIGHT of them in RAM beside this prop's decode.
                self._pump_tails(until=lambda: self._in_flight() < IN_FLIGHT)
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
        """Load the group's networks; after a failure everything is dropped and it is tried once more.

        The cap is set for this part first: the one in force is whatever the prop before left - lifted, because
        it had to spill, or the small one "finish" sets for writing a file - and under that a flow network of a
        few gigabytes could not be read at all.
        """
        loader = self.ctx.loader
        loader.set_cap(self.plan.NATIVE_RESERVE[part])
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
        exception, its traceback and the part function's frames are gone. What the part returned is in
        `_returned` (a FinishSteps from "finish", else None)."""
        self._returned = None
        try:
            self._returned = self.plan.TABLE[part](self.ctx, prop, mode)
        except (Cancelled, BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            return classify(exc), describe(exc), traceback.format_exc()
        return None

    def _after_failure(self, failure: tuple[str, str, str]) -> None:
        """Between two attempts of the ladder: the cache given back, what went ahead for the next prop too, and
        the card probed when the failure's words say its context may be gone. What was kept on the card for a later
        prop goes first: the retry gets the card it would have had without it."""
        loader = self.ctx.loader
        self._drop_extra()
        loader.trim()
        self.ctx.store.pump(upload=False)       # the next prop's inputs went ahead; the retry needs the room
        if failure[0] == "oom" and self._let_go():
            loader.trim()
        if failure[0] == "error" and any(words in failure[1].lower() for words in BROKEN_CONTEXT):
            loader.clear_cap()                  # the probe asks whether the card works, not whether the cap fits
            if not probe(self.ctx.device):
                raise BatchBroken(failure[1])

    def _prop_part(self, part: str, prop: PropState, position: int, count: int) -> None:
        ctx, loader, store = self.ctx, self.ctx.loader, self.ctx.store
        prop.state, prop.part = "working", part
        ctx.emit(Event("prop_start", part, prop.id, detail={"position": position, "of": count}))
        started = time.perf_counter()
        failure = None
        modes = self._modes()
        try:
            for mode in modes:
                if mode == "uncapped":
                    loader.clear_cap()
                    prop.slow = [*(name for name in prop.slow if name != part), part]
                    ctx.emit(Event("prop_spilled", part, prop.id))
                elif part != "finish":                  # "finish" sets its own cap, once per phase
                    loader.set_cap(self.plan.NATIVE_RESERVE[part])
                ooms = num_ooms(ctx.device)
                rng = torch.get_rng_state() if self.check_rng and part not in self.plan.DRAWING_PARTS else None
                failure = self._attempt(part, prop, mode)
                if failure is None and part in self.plan.CONV_PARTS and num_ooms(ctx.device) > ooms:
                    failure = ("oom", "an allocation failed inside a convolution", "")
                    self._discard(prop, self.plan.produces(part))       # nobody has read them yet
                if failure is None and rng is not None and not torch.equal(torch.get_rng_state(), rng):
                    failure = ("error", f"part {part} drew from the CPU random generator, which it must not", "")
                if failure is None:
                    break
                self._after_failure(failure)
                if failure[0] != "oom" or mode == modes[-1]:
                    break
        finally:
            ctx.inputs.clear()
            prop.seconds += time.perf_counter() - started
        if failure is not None:
            self._fail(prop, part, failure)
            return
        self._discard(prop, self.plan.discards(part, prop))
        if part in self.plan.TRIMMED_PARTS:
            loader.trim()
        if part != "finish":
            prop.state = "resting"
            ctx.emit(Event("prop_rest", part, prop.id))
            return
        if isinstance(self._returned, FinishSteps):
            self._tails.append(_Tail(prop, self._returned))
            self._returned = None
            self._pump_tails(until=lambda: True)        # its first GPU steps, while its mesh is fresh
            return
        self._done(prop)

    def _done(self, prop: PropState) -> None:
        prop.result = {**(prop.result or {}), "slow": list(prop.slow), "seconds": prop.seconds,
                       "resolution": prop.resolution, "resumed": prop.resumed}
        self.ctx.store.close_prop(prop.id)
        prop.state = "done"
        self.ctx.emit(Event("prop_done", "finish", prop.id, detail={"result": prop.result, "seconds": prop.seconds}))

    # ---- the finish steps -----------------------------------------------------------------------------------------

    def _pump_tails(self, until: Callable[[], bool]) -> None:
        """Move every prop that is in its finish steps along: a CPU step that is over is collected, CPU steps go
        to the worker threads, then GPU steps run here - every tail's CPU step is handed over before any GPU step
        runs, so no worker waits for a GPU step of another prop. Then, while `until()` is false, wait for
        the CPU steps and go on; `until` is true at once for a look that does not wait."""
        while True:
            fresh = self._fresh_tails()
            for tail in list(self._tails):
                if tail not in fresh:
                    self._advance(tail, gpu=False)
            for tail in list(self._tails):
                if tail not in fresh:
                    self._advance(tail, gpu=True)
            for tail in fresh:
                if tail in self._tails:
                    self._advance(tail, gpu=False)
            if until() or not self._tails:
                return
            running = [tail.future for tail in self._tails if tail.future is not None]
            if not running:
                return                              # nothing is under way: `until` cannot come true by waiting
            concurrent.futures.wait(running, timeout=STEP_WAIT, return_when=concurrent.futures.FIRST_COMPLETED)

    def _fresh_tails(self) -> list[_Tail]:
        """The tails that have run no step yet and begin with a CPU step, while props of the finish part are
        still to be decoded. Their first step is handed to the worker after this pump's GPU steps, not before
        them: what it makes (the hub's cuBVH tree of the freshly decoded mesh, a gigabyte in RAM for a 1536
        prop) is needed one texture later, while the GPU steps that run now let go of an older prop's; so two
        trees are in RAM at that moment, not three. The last prop's tail has nothing to wait for and is not
        deferred; nor is any tail whose first step takes the card."""
        if not any(prop.part != "finish" for prop in self._alive("finish", self._wave_now)):
            return []
        return [tail for tail in self._tails
                if tail.index == 0 and tail.future is None and tail.steps.steps
                and tail.steps.steps[0][0] == "cpu"]

    def _in_flight(self) -> int:
        """The tails whose steps still hold their prop's mesh and textures in RAM: every tail but those in their
        last step (the hub's export, which holds little beside the file it writes)."""
        return sum(1 for tail in self._tails if tail.index < len(tail.steps.steps) - 1)

    def _advance(self, tail: _Tail, gpu: bool = True) -> None:
        """Collect the tail's CPU step when it is over, hand its next CPU step to the worker, and with `gpu` run
        its GPU steps until it reaches a CPU step, its end, or a failure (without `gpu`, stop at a GPU step)."""
        steps = tail.steps
        if tail.future is not None:
            if not tail.future.done():
                return
            error = tail.future.exception()
            tail.future = None
            tail.open = False
            if error is not None:
                failure = (classify(error), describe(error), "".join(traceback.format_exception(error)))
                del error
                self._end_tail(tail, failure)
                return
            tail.index += 1
            self._step_done(tail)
        while tail.index < len(steps.steps):
            kind, fn = steps.steps[tail.index]
            if kind == "gpu" and not gpu:
                return
            tail.offered = steps.state["room_mib"] = self._room(tail)
            if kind == "cpu":
                if self._worker is None:
                    self._worker = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS,
                                                                         thread_name_prefix="p3d-finish-cpu")
                tail.open = True
                tail.future = self._worker.submit(fn, steps.state)
                return
            failure = self._gpu_step(tail, fn)
            if failure is not None:
                self._end_tail(tail, failure)
                return
            tail.index += 1
            self._step_done(tail)
        failure = self._result(tail)
        self._end_tail(tail, failure)

    def _gpu_step(self, tail: _Tail, fn: Callable[[dict], None]) -> tuple[str, str, str] | None:
        """One GPU step of a tail, down the ladder like a prop-part: under the hook's cap, after a trim, with
        the cap lifted when a spill is allowed."""
        ctx, loader, prop = self.ctx, self.ctx.loader, tail.prop
        modes = self._modes()
        failure = None
        for mode in modes:
            if mode == "uncapped":
                loader.clear_cap()
                prop.slow = [*(name for name in prop.slow if name != "finish"), "finish"]
                ctx.emit(Event("prop_spilled", "finish", prop.id))
            else:
                loader.set_cap(tail.steps.native_mib, safety_mib=tail.steps.safety_mib)
            started = time.perf_counter()
            failure = self._call(fn, tail.steps.state)
            prop.seconds += time.perf_counter() - started
            if failure is None:
                return None
            self._after_failure(failure)
            if failure[0] != "oom" or mode == modes[-1]:
                break
        return failure

    def _room(self, tail: _Tail) -> int:
        """MiB the next step of `tail` may keep on the card when it returns (plan.tail_room): bounded by the props
        of the finish part that are still to be decoded, and by what the other tails hold."""
        room = getattr(self.plan, "tail_room", None)
        if room is None:
            return 0
        pending = [prop for prop in self._alive("finish", self._wave_now) if prop.part != "finish"]
        held = sum(self._holding(other) for other in self._tails if other is not tail) + self._extra_mib()
        try:
            return int(room(self.ctx, pending, held))
        except Exception:
            return 0

    @staticmethod
    def _holding(tail: _Tail) -> int:
        """MiB a tail holds on the card, or is about to: what its state says, or - while a CPU step it was
        offered room runs, before any step of the tail has held - the most a step takes, since a step that will
        hold may not have written it yet. Once a step has held, the state is trusted: a later CPU step that says
        nothing (an export after the bake let go) holds nothing."""
        said = int(tail.steps.state.get("card_mib") or 0)
        coming = min(tail.offered, int(tail.steps.holds_mib or 0)) if tail.open and not tail.held else 0
        return max(said, coming)

    def _let_go(self) -> bool:
        """Every tail lets go of what it holds on the card (a decode ran out of memory). True when one did."""
        gone = False
        for tail in self._tails:
            steps = tail.steps
            if steps.release is None or not steps.state.get("card_mib"):
                continue
            with contextlib.suppress(Exception):
                steps.release(steps.state)
            steps.state.pop("card_mib", None)
            gone = True
        return gone

    def _result(self, tail: _Tail) -> tuple[str, str, str] | None:
        """The prop's result from its finished steps."""
        found: list[dict] = []
        failure = self._call(lambda state: found.append(dict(tail.steps.result(state) or {})), tail.steps.state)
        if failure is None:
            tail.prop.result = found[0]
        return failure

    @staticmethod
    def _call(fn: Callable[[dict], Any], state: dict) -> tuple[str, str, str] | None:
        """fn(state) on this thread; a failure as (kind, words, traceback text), the exception gone."""
        try:
            fn(state)
        except (BatchBroken, KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            return classify(exc), describe(exc), traceback.format_exc()
        return None

    def _step_done(self, tail: _Tail) -> None:
        """A step of the tail is over: the hook's steps count on after the three of the decode phase; the event
        says what room the step was offered and what the prop holds on the card now."""
        held = tail.steps.state.get("card_mib")
        if held:
            tail.held = True
        total = 3 + len(tail.steps.steps)
        self.ctx.emit(Event("step", "finish", tail.prop.id, 3 + tail.index, total,
                            detail={"room_mib": tail.offered, "card_mib": int(held or 0)}))

    def _end_tail(self, tail: _Tail, failure: tuple[str, str, str] | None) -> None:
        self._tails.remove(tail)
        tail.steps.state.clear()                # the mesh and the textures: RAM back at once
        if failure is not None:
            self._fail(tail.prop, "finish", failure)
        else:
            self._done(tail.prop)

    def _discard(self, prop: PropState, keys: tuple[str, ...]) -> None:
        if keys:
            self.ctx.store.discard(prop.id, *keys)

    def _fail(self, prop: PropState, part: str, failure: tuple[str, str, str]) -> None:
        kind, detail, trace = failure
        print(f"[staged] prop {prop.id} failed{' in ' + part if part else ''} ({kind}): {detail}", flush=True)
        if trace:
            print(trace, flush=True)
        self.ctx.store.close_prop(prop.id)
        self._drop_extra()
        self.ctx.loader.trim()
        prop.state = "failed"
        prop.error = {"kind": kind, "detail": detail}
        self.ctx.emit(Event("prop_failed", part, prop.id, detail=dict(prop.error)))
