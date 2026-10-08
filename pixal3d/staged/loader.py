"""Networks brought from disk straight onto the card when a part needs them, and dropped when it is done.

No copy of a network rests in RAM. A flow model or a decoder is built without its weights (nothing is initialised
that the file overwrites anyway), its file is read tensor by tensor through one 64 MiB buffer, each tensor is
uploaded, cast on the card to the dtype the model keeps it in, and assigned. The four small models (background
removal, MoGe, DINOv3, NAF) come from their own builders (builders.py). The loader also owns the cap on what
PyTorch may reserve on the card, and a background thread that reads the next network's file so that the operating
system has it in its file cache when its turn comes.

    loader = NetworkLoader("cuda", on_event=print)
    loader.register_checkpoint("hr_flow.sv", ckpts / "slat_flow_img2shape_dit_1_3B_1024_bf16")
    loader.register_builder("dino", builders["dino"], file=dino_folder / "model.safetensors")
    loader.set_cap()                                # before each prop: the budget follows the card's free memory
    model = loader.load("hr_flow.sv", cancelled)    # on the card, eval(), requires_grad off
    loader.prefetch("tex_flow.sv")                  # what comes next, read in the background
    ...
    del model
    loader.drop("hr_flow.sv")                       # nothing of it is left anywhere

Threads: everything is called on the one thread that owns the card, except `prefetch` and `close`. The prefetch
thread touches neither CUDA nor torch.

Events (`on_event`, called on the loading thread, must not block), all plain dicts:
    {"kind": "net_load", "name", "bytes_read", "bytes"}    when a load starts, then at most once a second
    {"kind": "net_ready", "name", "seconds", "bytes"}      the network is on the card
    {"kind": "net_drop", "name", "seconds", "bytes"}       it is gone again
`bytes` is what the load reads from disk (a builder without a file: the size of the built module).
"""

from __future__ import annotations

import contextlib
import gc
import itertools
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from . import stfile
from .types import Cancelled

MIB = 1 << 20
BUFFER_BYTES = 64 * MIB             # the one buffer every file is read through; a power of two, see _staging_buffer
PREFETCH_BYTES = 16 * MIB           # the prefetch thread's own read size
PREFETCH_SPARE_BYTES = 1536 * MIB   # RAM that must stay available beside a file before it is read ahead
CONTEXT_MIB = 300                   # what a CUDA context takes on the card outside PyTorch's allocator
LOADS_KEPT = 256                    # stats() remembers this many loads

# Checkpoints that lack tensors their model's constructor computes: class name -> those keys. Such a class is
# built for real (with the initialisers switched off) so that these tensors hold what the constructor gives them,
# as they do when the reference loads the file with strict=False. None of Pixal3D's eleven files needs it.
COMPUTED: dict[str, frozenset[str]] = {}

_staging: dict[bool, torch.Tensor] = {}
_staging_pinned_bytes = 0


@dataclass
class _Checkpoint:
    config_file: Path
    file: Path


@dataclass
class _Builder:
    build: Callable[[str], nn.Module]
    file: Path | None


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

    # ------------------------------------------------------------------------------------------------ registration

    def register_checkpoint(self, name: str, ckpt: Path) -> None:
        """A flow model or decoder: `ckpt` is the path without extension (<ckpt>.json and <ckpt>.safetensors).
        Nothing is read here: files that are missing or broken are reported by `load`."""
        self._free_name(name)
        self._sources[name] = _Checkpoint(Path(f"{ckpt}.json"), Path(f"{ckpt}.safetensors"))

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
            if module is not None:
                module.eval()
                module.requires_grad_(False)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
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
        MiB PyTorch may reserve from now on."""
        if self.device.type != "cuda":
            return 0
        safety = self.safety_mib if safety_mib is None else safety_mib
        free, total = torch.cuda.mem_get_info(self.device)
        budget = free + torch.cuda.memory_reserved(self.device) - (safety + native_reserve_mib) * MIB
        if self.max_vram_mib:
            budget = min(budget, (self.max_vram_mib - CONTEXT_MIB - safety - native_reserve_mib) * MIB)
        fraction = min(max(budget / total, 0.05), 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, self.device)
        self._cap_mib = int(fraction * total) // MIB
        return self._cap_mib

    def clear_cap(self) -> None:
        """Lift the limit: PyTorch may grow as far as the driver lets it."""
        if self._cap_mib is not None:
            torch.cuda.set_per_process_memory_fraction(1.0, self.device)
        self._cap_mib = None

    def capped(self) -> bool:
        return self._cap_mib is not None

    def loaded(self) -> list[str]:
        return list(self._modules)

    def stats(self) -> dict:
        """{"pinned_bytes": page-locked RAM of the read buffer, "cap_mib": the limit set_cap gave (None: none),
        "loads": [{"name", "seconds", "bytes"}] of the loads so far, "prefetch": what the background reader is
        at ({"name", "state", "bytes_read", "bytes"}) or None}."""
        return {"pinned_bytes": _staging_pinned_bytes, "cap_mib": self._cap_mib, "loads": list(self._loads),
                "prefetch": self._prefetcher.status() if self._prefetcher is not None else None}

    # ------------------------------------------------------------------------------------------------- any thread

    def prefetch(self, name: str | None) -> None:
        """Start reading that network's file in the background, in place of whatever was being read; None stops.
        The bytes are thrown away: the point is that the operating system keeps the file in its cache while RAM
        is free. Nothing is read for a network that is loaded or has no file."""
        if self._prefetcher is None:
            return
        file = None
        if name is not None:
            file = self._source(name).file
            if name in self._modules:
                file = None
        self._prefetcher.read(name, file)

    def close(self) -> None:
        """Stop the prefetch thread. Loaded networks stay."""
        if self._prefetcher is not None:
            self._prefetcher.close()

    # ---------------------------------------------------------------------------------------------------- inside

    def _source(self, name: str) -> _Checkpoint | _Builder:
        try:
            return self._sources[name]
        except KeyError:
            raise KeyError(f"no network is registered as {name!r} (registered: {sorted(self._sources)})") from None

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
        collected in `state`, which the caller owns and empties."""
        config = _config(source.config_file)
        header = stfile.read_header(source.file)
        cls, args = _model_class(config["name"]), config["args"]
        computed = COMPUTED.get(cls.__name__, frozenset())
        model = _build_for_real(cls, args) if computed else _build_empty(cls, args)
        want = {key: (value.dtype, tuple(value.shape)) for key, value in model.state_dict().items()}

        missing = sorted(want.keys() - header.keys() - computed)
        if missing:
            raise KeyError(f"{source.file} lacks {len(missing)} of the tensors {cls.__name__} needs "
                           f"(first: {missing[0]})")
        keys = [key for key in header if key in want]       # what the model has no place for is not read
        for key in keys:
            if header[key].shape != want[key][1]:
                raise ValueError(f"{source.file}: {key} is {list(header[key].shape)}, {cls.__name__} expects "
                                 f"{list(want[key][1])}")
        size = sum(header[key].nbytes for key in keys)
        self._emit("net_load", name=name, bytes_read=0, bytes=size)

        buffer = _staging_buffer(self.device.type == "cuda")
        done, told = 0, time.monotonic()
        reader = stfile.iter_tensors(source.file, buffer, keys)
        try:
            for key, info, data in reader:
                if isinstance(data, torch.Tensor):
                    tensor = data.to(self.device, copy=True)        # blocking: the buffer is free again at once
                else:
                    flat = torch.empty(info.nbytes, dtype=torch.uint8, device=self.device)
                    at = 0
                    for piece in data:
                        flat[at:at + piece.numel()].copy_(piece)
                        at += piece.numel()
                    tensor = stfile.as_tensor(flat, info)
                if tensor.dtype != want[key][0]:
                    tensor = tensor.to(want[key][0])                # cast on the device (the flows: fp32 to bf16)
                state[key] = tensor
                done += info.nbytes
                now = time.monotonic()
                if now - told >= 1.0:
                    told = now
                    self._emit("net_load", name=name, bytes_read=done, bytes=size)
                if cancelled is not None and cancelled():       # before the next tensor is read from the file
                    return None, 0
        finally:
            reader.close()

        model.load_state_dict(state, strict=False, assign=True)
        if not computed and _on_meta(model):
            # The constructor made something on the parameters' device that the file does not carry.
            model = _build_for_real(cls, args)
            model.load_state_dict(state, strict=False, assign=True)
        stranded = _on_meta(model)
        if stranded:
            raise RuntimeError(f"{cls.__name__}: {stranded[0]} has no values after loading {source.file}")
        model.to(self.device)                                       # what the constructor computed, if anything
        return model, size


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


def _staging_buffer(pinned: bool) -> torch.Tensor:
    """The process's one read buffer. Page-locked when the target is the card, so that uploads from it are plain
    copies; allocated once and kept for the life of the process, because torch rounds a pinned block up to a power
    of two and never returns it to the system, and because allocating one is a CUDA call that belongs to the
    thread that owns the card. When the system refuses to lock the pages the buffer is ordinary memory (the same
    values, a little slower)."""
    global _staging_pinned_bytes
    buffer = _staging.get(pinned)
    if buffer is None:
        if pinned:
            try:
                buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8, pin_memory=True)
                _staging_pinned_bytes = BUFFER_BYTES
            except RuntimeError:
                buffer = None
        if buffer is None:
            buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8)
        _staging[pinned] = buffer
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
