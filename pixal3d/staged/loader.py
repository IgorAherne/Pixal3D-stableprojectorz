"""Networks brought from disk straight onto the card when a part needs them, and dropped when it is done.

No copy of a network rests in RAM. A flow model or a decoder is built without its weights (nothing is initialised
that the file overwrites anyway), its file is read in file order, each tensor is uploaded, cast on the card to the
dtype the model keeps it in, and assigned. Onto the card the file is read by READERS threads at once, 16 MiB at a
time into slots of page-locked RAM (the 64 MiB read buffer and a pool of as much again beside it), and the uploads
are queued behind each other while the next bytes are read; every tensor lands in a block of the size and the
order the one-buffer way gives it, with the same bytes. Onto the CPU (tests) the file goes through the 64 MiB
buffer one tensor at a time. Where the installer has made a copy of
the file with every tensor already in the model's dtype (weights.py: the flows' fp32 tensors as bf16, half the
bytes), that copy is read instead and nothing is cast - as long as it was made from the file as it is now, by this
torch, and holds what the model takes; anything else about it (missing, stale, unreadable, breaking half-way) only
means that the original is read. The four small models (background removal, MoGe, DINOv3, NAF) come from their own
builders (builders.py). The loader also owns the cap on what PyTorch may reserve on the card, and a background
thread that reads the next network's file (its copy, when that is current) so that the operating system has it in
its file cache when its turn comes.

    loader = NetworkLoader("cuda", on_event=print)
    loader.register_checkpoint("hr_flow.sv", ckpts / "slat_flow_img2shape_dit_1_3B_1024_bf16")
    loader.register_builder("dino", builders["dino"], file=dino_folder / "model.safetensors")
    loader.set_cap()                                # before each prop: the budget follows the card's free memory
    model = loader.load("hr_flow.sv", cancelled)    # on the card, eval(), requires_grad off
    loader.prefetch("tex_flow.sv")                  # what comes next, read in the background
    ...
    del model
    loader.drop("hr_flow.sv")                       # nothing of it is left anywhere

Threads: everything is called on the one thread that owns the card, except `prefetch`. The prefetch thread and
the readers of a load (which live as long as the load) touch neither CUDA nor torch.

Events (`on_event`, called on the loading thread, must not block), all plain dicts:
    {"kind": "net_load", "name", "bytes_read", "bytes"}    when a load starts, then at most once a second
    {"kind": "net_ready", "name", "seconds", "bytes"}      the network is on the card
    {"kind": "net_drop", "name", "seconds", "bytes"}       it is gone again
`bytes` is what the load reads from disk: the tensors it takes from the checkpoint's copy or from the original,
whichever it reads (a builder without a file: the size of the built module).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import itertools
import json
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from . import stfile, weights
from .types import Cancelled

MIB = 1 << 20
BUFFER_BYTES = 64 * MIB             # the one buffer every file is read through; a power of two, see _staging_buffer.
#                                     It also decides how a tensor goes onto the device: one larger than it is
#                                     assembled piece by piece in a block of its own, a smaller one that is cast
#                                     passes through the one block taken for the casts
READERS = 3                         # threads that read a checkpoint's file at once for a load onto the card
#                                     (_read_parallel); 1 = the one-buffer way below, as before. Measured (RTX 3060,
#                                     a flow's 2.8 GB from the file cache): 0.71-0.82 s one way, 0.40-0.47 s this
#                                     way with 2 to 6 readers alike (the cache gave ~11 GB/s), a third of it the
#                                     model's construction on the meta device
CHUNK_BYTES = 16 * MIB              # what one reader reads at a time, into one slot of the page-locked pool
POOL_BYTES = 64 * MIB               # page-locked RAM beside the read buffer for the readers' slots (the read buffer
#                                     is cut into slots as well): made at the first load onto the card, given back
#                                     to the system by close()
PARALLEL_ON_CPU = False             # tests: loads onto the CPU take the parallel way as well
PREFETCH_BYTES = 16 * MIB           # the prefetch thread's own read size
PREFETCH_SPARE_BYTES = 1536 * MIB   # RAM that must stay available beside a file before it is read ahead
CONTEXT_MIB = 300                   # what a CUDA context takes on the card outside PyTorch's allocator
LOADS_KEPT = 256                    # stats() remembers this many loads

# Checkpoints that lack tensors their model's constructor computes: class name -> those keys. Such a class is
# built for real (with the initialisers switched off) so that these tensors hold what the constructor gives them,
# as they do when the reference loads the file with strict=False. None of Pixal3D's eleven files needs it.
COMPUTED: dict[str, frozenset[str]] = {}

_staging: dict[bool, torch.Tensor] = {}     # the process's one read buffer, under True once the card has asked
_staging_pinned_bytes = 0
_driver_cards: dict[int, Any] = {}  # card index -> (NVML library, its handle of the card), None: cannot be asked


@dataclass
class _Checkpoint:
    config_file: Path
    file: Path
    convert: Callable[[nn.Module], nn.Module] | None = None     # applied to the loaded module (modules/int8.py)


@dataclass
class _Builder:
    build: Callable[[str], nn.Module]
    file: Path | None


@dataclass
class _Plan:
    """A checkpoint's load as settled before any of its tensors is read (weights.py makes its copies from it too)."""
    cls: type
    args: dict
    computed: frozenset[str]        # what the class's constructor computes and no file carries (COMPUTED)
    model: nn.Module                # built without weights, or for real when the class is in COMPUTED
    header: dict[str, stfile.TensorInfo]                    # the checkpoint's own file
    want: dict[str, tuple[torch.dtype, tuple[int, ...]]]    # the model's state: each tensor's dtype and shape
    keys: list[str]                 # the file's tensors the model has a place for, in file order


class NetworkLoader:
    """The networks of one process: which exist (registered), which are on the device now (loaded), and how much
    PyTorch may reserve there (the cap)."""

    def __init__(self, device: str = "cuda", *, prefetch: str = "cache", max_vram_mib: int | None = None,
                 safety_mib: int = 300, on_event: Callable[[dict], None] | None = None):
        """`prefetch`: "cache" reads the next network's file ahead in the background, "off" never does.
        `max_vram_mib`: pretend the card has only this much (tests; a real smaller card also needs a block of
        memory held inside the process, because native allocations do not obey PyTorch's limit).
        `safety_mib`: what set_cap leaves free on the card for everybody else."""
        if prefetch not in ("off", "cache"):
            raise ValueError(f'prefetch is "off" or "cache", not {prefetch!r}')
        self.device = torch.device(device)
        self.max_vram_mib = max_vram_mib
        self.safety_mib = safety_mib
        self.on_event = on_event
        self._device_text = str(device)
        self._sources: dict[str, _Checkpoint | _Builder] = {}
        self._modules: dict[str, nn.Module] = {}
        self._bytes: dict[str, int] = {}
        self._loads: deque[dict] = deque(maxlen=LOADS_KEPT)
        self._cap_mib: int | None = None
        self._prefetcher = _Prefetcher() if prefetch == "cache" else None
        self._pool: torch.Tensor | None = None      # the readers' page-locked RAM beside the read buffer
        self._pool_pinned = 0
        self._card_bytes: dict[str, int] = {}       # loaded network -> bytes its parameters and buffers take

    # ------------------------------------------------------------------------------------------------ registration

    def register_checkpoint(self, name: str, ckpt: Path,
                            convert: Callable[[nn.Module], nn.Module] | None = None) -> None:
        """A flow model or decoder: `ckpt` is the path without extension (<ckpt>.json and <ckpt>.safetensors; a
        copy of the latter in the model's dtypes, weights.copy_of, is read in its place while it is valid).
        Nothing is read here: files that are missing or broken are reported by `load`. `convert`, when given, is
        applied to the module once its weights are on the device and before its size is measured (the flows'
        linears as int8 GEMMs, modules/int8.py: it returns the module unchanged when that is off)."""
        self._free_name(name)
        self._sources[name] = _Checkpoint(Path(f"{ckpt}.json"), Path(f"{ckpt}.safetensors"), convert)

    def register_builder(self, name: str, build: Callable[[str], nn.Module], file: Path | None = None) -> None:
        """A model with a loader of its own: `build(device)` returns it on that device. `file` is what `prefetch`
        reads ahead for it (its weights file), when it has one."""
        self._free_name(name)
        self._sources[name] = _Builder(build, Path(file) if file is not None else None)

    # ---------------------------------------------------------------------------------- the card's thread only

    def load(self, name: str, cancelled: Callable[[], bool] | None = None) -> nn.Module:
        """The network on the card, in eval mode, requires_grad off; the same object while it stays loaded.
        `cancelled` is asked between the tensors of a checkpoint: when it says yes, what was read is dropped and
        Cancelled is raised. A load that fails for any reason leaves nothing on the card."""
        module = self._modules.get(name)
        if module is not None:
            return module
        source = self._source(name)
        if cancelled is not None and cancelled():
            raise Cancelled()
        self.trim()                                     # weights go onto an empty card: where a block lands is final
        began = time.perf_counter()
        state: dict[str, torch.Tensor] = {}
        try:
            if isinstance(source, _Builder):
                module, size = self._build(name, source)
            else:
                module, size = self._read(name, source, state, cancelled)
                if module is not None and source.convert is not None:
                    module = source.convert(module)
            if module is not None:
                module.eval()
                module.requires_grad_(False)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()            # the blocks the file's fp32 tensors passed through
        except BaseException as error:
            module = None
            state.clear()
            traceback.clear_frames(error.__traceback__)     # the frames below still hold the tensor being read
            self.trim()
            raise
        state.clear()
        if module is None:
            self.trim()
            raise Cancelled()
        seconds = round(time.perf_counter() - began, 3)
        self._modules[name] = module
        self._bytes[name] = size
        self._card_bytes[name] = _tensor_bytes(module)
        self._loads.append({"name": name, "seconds": seconds, "bytes": size})
        self._emit("net_ready", name=name, seconds=seconds, bytes=size)
        return module

    def get(self, name: str) -> nn.Module | None:
        """The loaded network of that name, or None."""
        return self._modules.get(name)

    def drop(self, *names: str) -> None:
        """Forget these networks and give their memory back to the card. The caller must hold no other reference
        to them: there is no copy to go back to, and a network somebody still holds stays on the card."""
        began = time.perf_counter()
        gone = [name for name in names if self._modules.pop(name, None) is not None]
        for name in gone:
            self._card_bytes.pop(name, None)
        self.trim()
        seconds = round(time.perf_counter() - began, 3)
        for name in gone:
            self._emit("net_drop", name=name, seconds=seconds, bytes=self._bytes.pop(name, 0))

    def drop_all(self) -> None:
        self.drop(*self.loaded())

    def trim(self) -> None:
        """Collect garbage and hand PyTorch's unused blocks back to the card."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def set_cap(self, native_reserve_mib: int = 200, safety_mib: int | None = None) -> int:
        """Limit what PyTorch may reserve to what the card has free right now, less `native_reserve_mib` (memory
        the coming work takes outside PyTorch's allocator) and the safety margin. PyTorch then reuses its cache
        before it grows and raises torch.OutOfMemoryError instead of spilling into shared memory. Returns the
        MiB PyTorch may reserve from now on.

        "Free" is the smaller of two figures. CUDA's own (mem_get_info) is, under Windows, what the system grants
        this process: it does not move when another program fills the card. The driver's (NVML, what nvidia-smi
        shows) counts every program. Where the driver cannot be asked, CUDA's figure stands alone."""
        if self.device.type != "cuda":
            return 0
        safety = self.safety_mib if safety_mib is None else safety_mib
        card = self._card()
        free, total = torch.cuda.mem_get_info(card)
        seen = _driver_free(card)
        if seen is not None:
            free = min(free, seen)
        budget = free + torch.cuda.memory_reserved(card) - (safety + native_reserve_mib) * MIB
        if self.max_vram_mib:
            budget = min(budget, (self.max_vram_mib - CONTEXT_MIB - safety - native_reserve_mib) * MIB)
        fraction = min(max(budget / total, 0.05), 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, card)
        self._cap_mib = int(fraction * total) // MIB
        return self._cap_mib

    def clear_cap(self) -> None:
        """Lift the limit: PyTorch may grow as far as the driver lets it."""
        if self._cap_mib is not None:
            torch.cuda.set_per_process_memory_fraction(1.0, self._card())
        self._cap_mib = None

    def capped(self) -> bool:
        return self._cap_mib is not None

    def headroom_mib(self) -> int | None:
        """MiB the card can still give this process, PyTorch's reserve counted in (the allocator reuses it before
        it grows): what set_cap budgets before it takes the reserves off, the driver's figure where that is lower,
        and under an emulated size what that size leaves beside what is allocated. None off the card."""
        if self.device.type != "cuda":
            return None
        card = self._card()
        free, _ = torch.cuda.mem_get_info(card)
        seen = _driver_free(card)
        if seen is not None:
            free = min(free, seen)
        room = (free + torch.cuda.memory_reserved(card)) // MIB
        if self.max_vram_mib:
            room = min(room, self.max_vram_mib - CONTEXT_MIB - torch.cuda.memory_allocated(card) // MIB)
        return int(room)

    def loaded(self) -> list[str]:
        return list(self._modules)

    def size_mib(self, name: str) -> int | None:
        """MiB the loaded network's parameters and buffers take on its device; None when it is not loaded."""
        size = self._card_bytes.get(name) if name in self._modules else None
        return None if size is None else -(-size // MIB)

    def stats(self) -> dict:
        """{"pinned_bytes": page-locked RAM of the read buffer and the readers' slots, "cap_mib": the limit
        set_cap gave (None: none), "loads": [{"name", "seconds", "bytes"}] of the loads so far, "prefetch": what the
        background reader is at ({"name", "state", "bytes_read", "bytes"}) or None}."""
        return {"pinned_bytes": _staging_pinned_bytes + self._pool_pinned, "cap_mib": self._cap_mib,
                "loads": list(self._loads),
                "prefetch": self._prefetcher.status() if self._prefetcher is not None else None}

    # ------------------------------------------------------------------------------------------------- any thread

    def prefetch(self, name: str | None) -> None:
        """Start reading that network's file in the background, in place of whatever was being read; None stops.
        The bytes are thrown away: the point is that the operating system keeps the file in its cache while RAM
        is free. For a checkpoint whose copy was made from its file as it is now, the copy is what a load will
        read, so it is the copy that is read ahead (deciding that takes a look at two headers, here). Nothing is
        read for a network that is loaded or has no file."""
        if self._prefetcher is None:
            return
        file = None
        if name is not None:
            source = self._source(name)
            file = source.file
            if name in self._modules:
                file = None
            elif isinstance(source, _Checkpoint):
                file = weights.current_copy(source.file) or file
        self._prefetcher.read(name, file)

    def close(self) -> None:
        """Stop the prefetch thread and give the readers' page-locked RAM back to the system. Loaded networks
        stay."""
        if self._prefetcher is not None:
            self._prefetcher.close()
        if self._pool is not None:
            self._pool = None
            self._pool_pinned = 0
            if torch.cuda.is_initialized() and hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()     # torch keeps a freed page-locked block for later: let go of it

    # ---------------------------------------------------------------------------------------------------- inside

    def _source(self, name: str) -> _Checkpoint | _Builder:
        try:
            return self._sources[name]
        except KeyError:
            raise KeyError(f"no network is registered as {name!r} (registered: {sorted(self._sources)})") from None

    def _card(self) -> int:
        """The index of the card (torch's memory-fraction call does not take a bare "cuda")."""
        return self.device.index if self.device.index is not None else torch.cuda.current_device()

    def _free_name(self, name: str) -> None:
        if name in self._modules:
            raise RuntimeError(f"{name} is loaded: drop it before registering it again")

    def _emit(self, kind: str, **fields: Any) -> None:
        if self.on_event is not None:
            self.on_event({"kind": kind, **fields})

    def _build(self, name: str, source: _Builder) -> tuple[nn.Module, int]:
        size = source.file.stat().st_size if source.file is not None else 0
        self._emit("net_load", name=name, bytes_read=0, bytes=size)
        module = source.build(self._device_text)
        if not size:
            size = sum(tensor.numel() * tensor.element_size()
                       for tensor in itertools.chain(module.parameters(), module.buffers()))
        return module, size

    def _read(self, name: str, source: _Checkpoint, state: dict[str, torch.Tensor],
              cancelled: Callable[[], bool] | None) -> tuple[nn.Module | None, int]:
        """A checkpoint onto the device: (the module, bytes read), or (None, 0) when cancelled. The tensors are
        collected in `state`, which the caller owns and empties. They come from the checkpoint's copy when it is
        valid (weights.usable_copy), else from its own file; a copy that fails while it is read is given up for
        the original, read from its first tensor."""
        plan = _plan(source.config_file, source.file)
        copy = weights.usable_copy(source.file, plan.keys, plan.want)
        if copy is not None:
            broken = False
            try:
                return self._read_file(name, plan, *copy, state, cancelled)
            except (OSError, EOFError) as error:
                # The original is read once this block is left: the traceback holds the tensors read so far.
                traceback.clear_frames(error.__traceback__)
                broken = True
            if broken:
                state.clear()
                self.trim()
        return self._read_file(name, plan, source.file, plan.header, state, cancelled)

    def _read_file(self, name: str, plan: _Plan, file: Path, header: dict[str, stfile.TensorInfo],
                   state: dict[str, torch.Tensor], cancelled: Callable[[], bool] | None
                   ) -> tuple[nn.Module | None, int]:
        """_read from one file: the checkpoint's own (`header` is then plan.header) or its copy."""
        cls, args, computed, model, want, keys = plan.cls, plan.args, plan.computed, plan.model, plan.want, plan.keys
        size = sum(header[key].nbytes for key in keys)
        self._emit("net_load", name=name, bytes_read=0, bytes=size)

        buffer = _staging_buffer(self.device.type == "cuda")
        # Tensors the model keeps in another dtype than the file's (the flows: fp32 to bf16) are cast on the
        # device. They all pass through ONE block there, taken before any weight: uploaded into blocks of their
        # own they would leave a hole behind every kept tensor (165 MiB of holes in a flow model). A copy holds
        # them in the model's dtypes already: no block is taken then.
        passing = max((header[key].nbytes for key in keys if header[key].torch_dtype != want[key][0]
                       and header[key].nbytes <= buffer.numel()), default=0)
        scratch = torch.empty(passing, dtype=torch.uint8, device=self.device) if passing else None
        if READERS > 1 and (self.device.type == "cuda" or PARALLEL_ON_CPU):
            slots = self._slots(buffer)
            if len(slots) >= 2:
                if not self._read_parallel(name, file, header, keys, want, buffer.numel(), scratch, slots, state,
                                           cancelled, size):
                    return None, 0
                return self._assembled(plan, file, state), size
        done, told = 0, time.monotonic()
        reader = stfile.iter_tensors(file, buffer, keys)
        try:
            for key, info, data in reader:
                cast = info.torch_dtype != want[key][0]
                if not isinstance(data, torch.Tensor):              # larger than the buffer: piece by piece
                    flat = torch.empty(info.nbytes, dtype=torch.uint8, device=self.device)
                    at = 0
                    for piece in data:
                        flat[at:at + piece.numel()].copy_(piece)
                        at += piece.numel()
                    tensor = stfile.as_tensor(flat, info)
                elif cast and info.nbytes:
                    tensor = stfile.as_tensor(scratch[:info.nbytes], info)
                    tensor.copy_(data)
                else:
                    tensor = data.to(self.device, copy=True)
                if cast:
                    tensor = tensor.to(want[key][0])
                state[key] = tensor                                 # the copies block: the buffer is free again
                done += info.nbytes
                now = time.monotonic()
                if now - told >= 1.0:
                    told = now
                    self._emit("net_load", name=name, bytes_read=done, bytes=size)
                if cancelled is not None and cancelled():       # before the next tensor is read from the file
                    return None, 0
        finally:
            reader.close()
        return self._assembled(plan, file, state), size

    def _slots(self, buffer: torch.Tensor) -> list[torch.Tensor]:
        """The readers' slots: CHUNK_BYTES-long pieces of the read buffer and of the pool beside it (page-locked
        when the read buffer is; made here the first time)."""
        if self._pool is None:
            pinned = buffer.is_pinned() if self.device.type == "cuda" else False
            pool = None
            if pinned:
                try:
                    pool = torch.empty(POOL_BYTES, dtype=torch.uint8, pin_memory=True)
                    self._pool_pinned = POOL_BYTES
                except RuntimeError:
                    pool = None
            self._pool = pool if pool is not None else torch.empty(POOL_BYTES, dtype=torch.uint8)
        return [whole[at:at + CHUNK_BYTES] for whole in (buffer, self._pool)
                for at in range(0, whole.numel() - CHUNK_BYTES + 1, CHUNK_BYTES)]

    def _read_parallel(self, name: str, file: Path, header: dict[str, stfile.TensorInfo], keys: list[str],
                       want: dict, buffer_bytes: int, scratch: torch.Tensor | None, slots: list[torch.Tensor],
                       state: dict[str, torch.Tensor], cancelled: Callable[[], bool] | None, size: int) -> bool:
        """The tensors of `keys` onto the device as the one-buffer way puts them there - each tensor in a block of
        the same size, taken in the same order, holding the same bytes - but the file is read by READERS threads
        at once, CHUNK_BYTES at a time into the slots, and the copies to the card are only queued (from page-locked
        slots they run while the next chunks are read; a slot is read into again once its copies have run). False
        when `cancelled` said yes (asked once per tensor, as the other way asks it). A reader's OSError or EOFError
        is raised here, after every copy has run."""
        cuda = self.device.type == "cuda"
        chunks = _chunks(header, keys, CHUNK_BYTES)
        readers = _Readers(file, chunks, slots, READERS, cuda)
        done, told = 0, time.monotonic()
        index, slot, base, length = -1, -1, 0, 0
        try:
            for key in keys:
                info = header[key]
                cast = info.torch_dtype != want[key][0]
                dest = None
                if info.nbytes > buffer_bytes:                      # assembled in a block of its own
                    target = flat = torch.empty(info.nbytes, dtype=torch.uint8, device=self.device)
                elif cast and info.nbytes:
                    target = scratch[:info.nbytes]
                else:
                    dest = torch.empty(info.shape, dtype=info.torch_dtype, device=self.device)
                    target = dest.reshape(-1).view(torch.uint8)
                at, where = 0, info.start
                while where < info.end:
                    if not base <= where < base + length:           # the next chunk holds the next bytes
                        if slot >= 0:
                            readers.give(slot)
                            slot = -1
                        index += 1
                        slot = readers.take(index)
                        base, length = chunks[index]
                        if not base <= where < base + length:
                            raise RuntimeError(f"{file}: {key} is not where the ranges read say")
                    count = min(info.end, base + length) - where
                    target[at:at + count].copy_(slots[slot][where - base:where - base + count], non_blocking=cuda)
                    at += count
                    where += count
                if dest is not None:
                    tensor = dest
                else:
                    tensor = stfile.as_tensor(flat if info.nbytes > buffer_bytes else scratch[:info.nbytes], info)
                if cast:
                    tensor = tensor.to(want[key][0])
                state[key] = tensor
                done += info.nbytes
                now = time.monotonic()
                if now - told >= 1.0:
                    told = now
                    self._emit("net_load", name=name, bytes_read=done, bytes=size)
                if cancelled is not None and cancelled():
                    return False
        finally:
            if slot >= 0:
                readers.give(slot)
            readers.close()             # the threads stopped, every copy out of a slot has run
        return True

    def _assembled(self, plan: _Plan, file: Path, state: dict[str, torch.Tensor]) -> nn.Module:
        """The module of `plan` with the tensors read into `state`."""
        cls, args, computed, model = plan.cls, plan.args, plan.computed, plan.model
        model.load_state_dict(state, strict=False, assign=True)
        if not computed and _on_meta(model):
            # The constructor made something on the parameters' device that the file does not carry.
            model = _build_for_real(cls, args)
            model.load_state_dict(state, strict=False, assign=True)
        stranded = _on_meta(model)
        if stranded:
            raise RuntimeError(f"{cls.__name__}: {stranded[0]} has no values after loading {file}")
        model.to(self.device)                                       # what the constructor computed, if anything
        return model


def _plan(config_file: Path, file: Path) -> _Plan:
    """A checkpoint's load settled: its config read, its model built without weights (for real when COMPUTED
    lists the class), its file's header checked against the model's state. KeyError when the file lacks a tensor
    the model needs, ValueError when one has another shape; nothing of the tensors is read."""
    config = _config(config_file)
    header = stfile.read_header(file)
    cls, args = _model_class(config["name"]), config["args"]
    computed = COMPUTED.get(cls.__name__, frozenset())
    model = _build_for_real(cls, args) if computed else _build_empty(cls, args)
    want = {key: (value.dtype, tuple(value.shape)) for key, value in model.state_dict().items()}

    missing = sorted(want.keys() - header.keys() - computed)
    if missing:
        raise KeyError(f"{file} lacks {len(missing)} of the tensors {cls.__name__} needs (first: {missing[0]})")
    keys = [key for key in header if key in want]       # what the model has no place for is not read
    for key in keys:
        if header[key].shape != want[key][1]:
            raise ValueError(f"{file}: {key} is {list(header[key].shape)}, {cls.__name__} expects "
                             f"{list(want[key][1])}")
    return _Plan(cls, args, computed, model, header, want, keys)


def _config(file: Path) -> dict:
    """A checkpoint's config: {"name": the model class, "args": its constructor's arguments}."""
    with open(file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or "name" not in config or not isinstance(config.get("args"), dict):
        raise ValueError(f"{file} does not name a model class and its arguments")
    return config


def _model_class(name: str) -> type:
    from .. import models

    return getattr(models, name)


def _build_empty(cls: type, args: dict) -> nn.Module:
    """The module with every parameter on the meta device: shapes and dtypes, no memory, nothing initialised.
    (Not `with torch.device("meta")`: that would also empty plain tensor attributes such as the rotary
    embedders' `freqs`, which no file restores.)"""
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        return cls(**args)


def _build_for_real(cls: type, args: dict) -> nn.Module:
    """The module built on the CPU as the reference builds it, except that the initialisers write nothing."""
    with _initialisers_off():
        return cls(**args)


@contextlib.contextmanager
def _initialisers_off() -> Iterator[None]:
    """torch.nn.init's in-place functions do nothing inside the block."""
    names = [name for name in dir(torch.nn.init)
             if name.endswith("_") and not name.startswith("_") and callable(getattr(torch.nn.init, name))]
    kept = {name: getattr(torch.nn.init, name) for name in names}
    for name in names:
        setattr(torch.nn.init, name, lambda tensor, *args, **kwargs: tensor)
    try:
        yield
    finally:
        for name, function in kept.items():
            setattr(torch.nn.init, name, function)


def _on_meta(model: nn.Module) -> list[str]:
    """Names of the parameters and buffers that have no memory."""
    return [name for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers())
            if tensor.is_meta]


def _tensor_bytes(module: nn.Module) -> int:
    """Bytes of the module's parameters and buffers."""
    return sum(tensor.numel() * tensor.element_size()
               for tensor in itertools.chain(module.parameters(), module.buffers()))


def _chunks(header: dict[str, stfile.TensorInfo], keys: list[str], chunk_bytes: int) -> list[tuple[int, int]]:
    """The bytes of the file the tensors of `keys` (in file order) take, as (start, length) ranges of at most
    `chunk_bytes`, in file order: neighbouring tensors share a range, a gap (a tensor that is not read) ends one."""
    spans: list[list[int]] = []
    for key in keys:
        info = header[key]
        if not info.nbytes:
            continue
        if spans and spans[-1][1] == info.start:
            spans[-1][1] = info.end
        else:
            spans.append([info.start, info.end])
    return [(at, min(chunk_bytes, end - at)) for start, end in spans for at in range(start, end, chunk_bytes)]


class _Readers:
    """Threads that read byte ranges of one file (`chunks`, in file order) into slots, for one load. A reader
    takes a free slot and the next range together, so the ranges in the slots are always the next ones the loading
    thread takes, which it does in order (take); it gives a slot back once it has queued the copies out of it
    (give). On the card a slot is read into again only after an event recorded behind those copies has completed.
    The threads touch neither CUDA nor torch: they read into memoryviews of the slots made here, on the loading
    thread. close() stops them and waits for every copy out of a slot."""

    def __init__(self, file: Path, chunks: list[tuple[int, int]], slots: list[torch.Tensor], threads: int,
                 cuda: bool):
        self._file = file
        self._chunks = chunks
        self._views = [memoryview(slot.numpy()) for slot in slots]
        self._cuda = cuda
        self._cv = threading.Condition()
        self._free: list[int] = list(range(len(slots)))
        self._next = 0
        self._ready: dict[int, int] = {}            # range -> the slot it was read into
        self._flight: deque[tuple[Any, int]] = deque()      # (event, slot) whose copies may still run: loading thread
        self._error: BaseException | None = None
        self._stop = False
        count = max(1, min(threads, len(chunks))) if chunks else 0
        self._threads = [threading.Thread(target=self._run, name="p3d-net-read", daemon=True) for _ in range(count)]
        for thread in self._threads:
            thread.start()

    def _run(self) -> None:
        try:
            with open(self._file, "rb", buffering=0) as handle:
                while True:
                    with self._cv:
                        while not self._stop and self._next < len(self._chunks) and not self._free:
                            self._cv.wait()
                        if self._stop or self._next >= len(self._chunks):
                            return
                        slot, index = self._free.pop(), self._next
                        self._next += 1
                    start, length = self._chunks[index]
                    handle.seek(start)
                    stfile._fill(handle, self._views[slot][:length], self._file)
                    with self._cv:
                        self._ready[index] = slot
                        self._cv.notify_all()
        except BaseException as error:              # OSError, EOFError: the loading thread raises it
            with self._cv:
                if self._error is None:
                    self._error = error
                self._stop = True
                self._cv.notify_all()

    def take(self, index: int) -> int:
        """The slot range `index` was read into, once it has been. Meanwhile slots whose copies have run go back to
        the readers, and when every slot waits for its copies the oldest are waited for."""
        if not 0 <= index < len(self._chunks):
            raise RuntimeError(f"{self._file}: range {index} of {len(self._chunks)} asked for")
        while True:
            self._collect(wait=False)
            with self._cv:
                if index in self._ready:
                    return self._ready.pop(index)
                if self._error is not None:
                    raise self._error
                if self._free or not self._flight:          # a reader is at it
                    self._cv.wait(0.005 if self._flight else None)
                    continue
            self._collect(wait=True)

    def give(self, slot: int) -> None:
        """The loading thread has queued every copy out of `slot`."""
        if self._cuda:
            event = torch.cuda.Event()
            event.record()
            self._flight.append((event, slot))
            return
        with self._cv:
            self._free.append(slot)
            self._cv.notify_all()

    def _collect(self, wait: bool) -> None:
        """Slots whose copies have run go back to the readers; with `wait`, the oldest after waiting for it."""
        freed: list[int] = []
        while self._flight:
            event, slot = self._flight[0]
            if wait and not freed:
                event.synchronize()
            elif not event.query():
                break
            self._flight.popleft()
            freed.append(slot)
        if freed:
            with self._cv:
                self._free.extend(freed)
                self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        for thread in self._threads:
            thread.join()
        while self._flight:
            self._flight.popleft()[0].synchronize()


class _DriverMemory(ctypes.Structure):
    """NVML's nvmlMemory_t, in bytes."""
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _driver_free(card: int) -> int | None:
    """Bytes free on the card as the driver counts them over all programs (NVML), or None when it cannot be
    asked: no NVML library, or more than one card and none with this card's UUID."""
    if card not in _driver_cards:
        _driver_cards[card] = _driver_card(card)
    if _driver_cards[card] is None:
        return None
    library, handle = _driver_cards[card]
    memory = _DriverMemory()
    if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
        return None
    return int(memory.free)


def _driver_card(card: int) -> tuple[Any, Any] | None:
    """NVML and its handle of the card CUDA numbers `card`: found by the card's UUID (the two number cards in
    different orders), or the only card there is."""
    try:
        if os.name == "nt":
            system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll"
            library = ctypes.WinDLL(str(system) if system.is_file() else "nvml.dll")
        else:
            library = ctypes.CDLL("libnvidia-ml.so.1")
        if library.nvmlInit_v2() != 0:
            return None
        handle, count = ctypes.c_void_p(), ctypes.c_uint(0)
        uuid = getattr(torch.cuda.get_device_properties(card), "uuid", None)
        if uuid is not None and library.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode("ascii"),
                                                                  ctypes.byref(handle)) == 0:
            return library, handle
        if (library.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0 and count.value == 1
                and library.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(handle)) == 0):
            return library, handle
    except (OSError, AttributeError):
        pass
    return None


def _staging_buffer(pinned: bool) -> torch.Tensor:
    """The process's one read buffer. Page-locked when the card asks for it, so that uploads from it are plain
    copies, and that one buffer then serves loads onto the CPU as well; allocated once and kept for the life of
    the process, because torch rounds a pinned block up to a power of two and never returns it to the system, and
    because allocating one is a CUDA call that belongs to the thread that owns the card. When the system refuses
    to lock the pages the buffer is ordinary memory (the same values, a little slower)."""
    global _staging_pinned_bytes
    buffer = _staging.get(True)
    if buffer is not None:
        return buffer
    if not pinned:
        buffer = _staging.get(False)
        if buffer is None:
            buffer = _staging[False] = torch.empty(BUFFER_BYTES, dtype=torch.uint8)
        return buffer
    try:
        buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8, pin_memory=True)
        _staging_pinned_bytes = BUFFER_BYTES
    except RuntimeError:
        buffer = _staging.get(False)
        if buffer is None:
            buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8)
    _staging.clear()                        # with it goes the buffer of an earlier load onto the CPU, if there was one
    _staging[True] = buffer
    return buffer


class _Prefetcher:
    """One daemon thread that reads a file from end to end and throws the bytes away, one file at a time. A new
    request replaces the running one between two reads."""

    def __init__(self) -> None:
        self._wake = threading.Condition()
        self._thread: threading.Thread | None = None
        self._request: tuple[str, Path] | None = None
        self._serial = 0
        self._closed = False
        self._status: dict | None = None

    def read(self, name: str | None, file: Path | None) -> None:
        with self._wake:
            if self._closed:
                return
            self._serial += 1
            self._request = (name, file) if name is not None and file is not None else None
            if self._request is not None and self._thread is None:
                self._thread = threading.Thread(target=self._run, name="p3d-net-prefetch", daemon=True)
                self._thread.start()
            self._wake.notify_all()

    def status(self) -> dict | None:
        with self._wake:
            return dict(self._status) if self._status is not None else None

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._serial += 1
            thread, self._thread = self._thread, None
            self._wake.notify_all()
        if thread is not None:
            thread.join()

    def _run(self) -> None:
        view = memoryview(bytearray(PREFETCH_BYTES))
        while True:
            with self._wake:
                while self._request is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                (name, file), serial = self._request, self._serial
                self._request = None
                self._status = {"name": name, "state": "reading", "bytes_read": 0, "bytes": 0}
            self._tell(state=self._read_through(file, serial, view))

    def _read_through(self, file: Path, serial: int, view: memoryview) -> str:
        """Read the file unless RAM is short: "done", "stopped" (another request came), "skipped" or "failed"."""
        try:
            import psutil

            size = file.stat().st_size
            self._tell(bytes=size)
            if psutil.virtual_memory().available < size + PREFETCH_SPARE_BYTES:
                return "skipped"
            done = 0
            with open(file, "rb", buffering=0) as handle:
                while True:
                    if self._serial != serial:
                        return "stopped"
                    count = handle.readinto(view)
                    if not count:
                        return "done"
                    done += count
                    self._tell(bytes_read=done)
        except (OSError, ImportError):
            return "failed"

    def _tell(self, **fields: Any) -> None:
        with self._wake:
            self._status.update(fields)
