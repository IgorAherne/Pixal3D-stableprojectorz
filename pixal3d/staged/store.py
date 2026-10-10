"""The per-prop store of the staged engine: what the parts of one prop hand to each other while the other props of
the batch go through the same part.

Small items (the request, the cut-out pictures, coords, latents, the random generator's state) are megabytes and
stay in RAM for the prop's life. Large items (DINOv3's tokens and the conditionings, 4-200 MiB each) come off the
card into a RAM cache with a budget; what the budget cannot hold is written to the batch's work folder and read back
when the prop's turn comes. The item the next prop needs is brought into a page-locked slot ahead of time and
uploaded from there without waiting, behind the kernels of the prop that is being sampled. One checkpoint per prop,
its final latents, is the only thing that survives the process: a later batch on the same folder resumes that prop
at its last part.

    store = PropStore(work_dir, "cuda", ram_mib=256, slot_mib=128)
    resumed = store.begin({prop_id: request_hash})          # {} unless an earlier run left a valid checkpoint
    store.open_prop(prop_id)
    store.put_small(prop_id, "coords_hr", {"coords": coords.cpu()})
    store.put_device(prop_id, "cond_hr", {"global": g, "proj": p})     # tensors on the card -> RAM (-> disk)
    store.want(prop_id, "cond_hr")                          # the prop before this one starts: stage it
    store.pump()                                            # every sampler step: upload it when there is room
    cond = store.take(prop_id, "cond_hr")                   # on the card; the caller owns these tensors
    store.discard(prop_id, "cond_hr")
    store.close_prop(prop_id)

Threads. Every method is called on the GPU thread (the launcher's one worker thread); `stats()` may be read from
anywhere. The helper thread `p3d-store-io` does all file work, in the order it was asked for, and copies bytes into
a slot it was lent. It makes and frees no tensor memory and never calls CUDA: torch's CPU allocator gives memory
back to Windows only from the thread that took it, and page-locking memory is a CUDA call. Whatever the helper
works on stays referenced by the GPU thread until the helper has reported back and the GPU thread has let go of it.

Page-locked slots. `slots` buffers of exactly `slot_mib` MiB are made at the first `want()` of the process and kept
for its life (a freed page-locked block is never given back to Windows). A slot goes free -> filling (helper
thread) -> staged -> uploading -> free, and it becomes free only when the CUDA event recorded behind its own copies
has completed, whatever happened to the item meanwhile: bytes still being copied to the card are never overwritten.

Files: <work_dir>/<prop_id>/<key>.safetensors for a spilled item, <work_dir>/<prop_id>/final.safetensors and
final.json for the checkpoint. The store deletes only files of these names; the work folder itself is the caller's.
"""

from __future__ import annotations

import collections
import ctypes
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil
import torch

from . import stfile
from .types import PropFailed

MIB = 1 << 20
GIB = 1 << 30
ALIGN = 64                          # every tensor of a staged item starts on a multiple of this inside its slot
UPLOAD_HEADROOM = 256 * MIB         # what an upload issued ahead of its prop's turn leaves free on the card
DISK_FLOOR = GIB                    # a spill must leave this much free on the drive
RAM_FLOOR = 3 * GIB // 2            # with less RAM available, an item that cannot be spilled fails its prop
DISK_MESSAGE = "Not enough free disk space in the project folder for Pixal3D's temporary files."
CHECKPOINT = "final"                # <prop>/final.safetensors + final.json
WAIT = 0.005                        # seconds a waiting take() / flush() sleeps between two rounds of pumping
PATIENCE = 300.0                    # seconds take() / flush() / begin() wait for the disk before they give up
CLOSE_PATIENCE = 30.0               # the same for close()
DELETE_WAITS = (0.4, 0.6, 1.0)      # a file Windows will not let go: three more tries over two seconds

FREE, FILLING, STAGED, UPLOADING = "free", "filling", "staged", "uploading"
_FORBIDDEN = '<>:"/\\|?*'

_spare: dict[tuple[int, bool], list[torch.Tensor]] = {}     # slot buffers of closed stores, by (bytes, for a card)
_spare_lock = threading.Lock()


def ram_defaults() -> tuple[int, int]:
    """(ram_mib, slot_mib) for this PC: a RAM cache of min(768, 5% of the RAM) MiB and slots of 256 MiB; on a PC
    with 16 GiB or less a cache of 256 MiB and slots of 128 MiB."""
    total = int(psutil.virtual_memory().total)
    if total <= 16 * GIB:
        return 256, 128
    return min(768, total // 20 // MIB), 256


def _free_bytes(folder: Path) -> int:
    """Free bytes on the drive of a folder."""
    return int(shutil.disk_usage(folder).free)


def _ram_available() -> int:
    """Bytes of RAM Windows can still hand out without the page file."""
    return int(psutil.virtual_memory().available)


def _plain(name: Any, what: str) -> str:
    """A prop id or the key of a large item as the name of a folder or file: one path component Windows accepts."""
    if (not isinstance(name, str) or not name or name in (".", "..") or name != name.strip() or name.endswith(".")
            or any(char in _FORBIDDEN or ord(char) < 32 for char in name)):
        raise ValueError(f"{name!r} cannot be a {what}: it becomes the name of a file")
    return name


def _ours(name: str) -> bool:
    """Is this the name of a file the store writes into a prop's folder."""
    return name.endswith((".safetensors", ".tmp")) or name == CHECKPOINT + ".json"


def _resting(tensor: torch.Tensor) -> torch.Tensor:
    """A tensor as the store keeps it in RAM: detached, contiguous, and always memory of the store's own - the
    helper thread writes a file straight out of it, for as long as that takes, while the caller goes on using what
    it handed in. A blocking copy for a tensor on the card; a copy for one already in RAM as well."""
    value = tensor.detach()
    if value.device.type == "cpu":
        return value.clone(memory_format=torch.contiguous_format)
    value = value.cpu()
    return value if value.is_contiguous() else value.contiguous()


def _aligned(size: int) -> int:
    return -(-size // ALIGN) * ALIGN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    block = bytearray(MIB)
    with open(path, "rb", buffering=0) as handle:
        while True:
            count = handle.readinto(block)
            if not count:
                return digest.hexdigest()
            digest.update(memoryview(block)[:count])


def _borrow(size: int, card: bool) -> torch.Tensor:
    """A slot buffer: one a closed store of this process gave back, else a new one - page-locked for a store that
    serves a card, pageable when Windows refuses to lock that much (copies from it then wait, same values)."""
    with _spare_lock:
        spare = _spare.get((size, card))
        if spare:
            return spare.pop()
    if card:
        try:
            return torch.empty(size, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(size, dtype=torch.uint8)


class _Refused(OSError):
    """File work the helper thread gave up on, in words for the warning as they are."""


class _Slot:
    """One slot buffer and where it is in its life cycle."""
    __slots__ = ("buf", "pinned", "state", "item", "views", "direct", "job", "event")

    def __init__(self, buf: torch.Tensor, pinned: bool):
        self.buf = buf                  # 1-D uint8, exactly slot_mib MiB
        self.pinned = pinned            # torch knows it as page-locked memory
        self.state = FREE
        self.item: _Item | None = None  # filling / staged: whom for (None: the item went another way meanwhile)
        self.views: dict[str, torch.Tensor] | None = None       # staged: name -> view of buf, typed and shaped
        self.direct = False             # staged: a copy from the views to the card does not wait
        self.job: _Job | None = None    # filling: the helper thread's job
        self.event: Any = None          # uploading: completes when the last copy out of buf has run


class _Item:
    """One large item of one prop: where its bytes are, and who is working on them."""
    __slots__ = ("prop", "key", "path", "names", "layout", "nbytes", "need", "ram", "spill", "stays_in_ram",
                 "on_disk", "file_bytes", "wanted", "skip_slot", "slot", "device", "upload", "touch")

    def __init__(self, prop: "_Prop", key: str, tensors: dict[str, torch.Tensor]):
        self.prop = prop
        self.key = key
        self.path = prop.folder / f"{key}.safetensors"
        self.names = list(tensors)                          # its tensors, in the order they were put
        self.layout = [(name, tensor.dtype, tuple(tensor.shape), tensor.numel() * tensor.element_size())
                       for name, tensor in tensors.items()]
        self.nbytes = sum(size for _, _, _, size in self.layout)
        self.need = sum(_aligned(size) for _, _, _, size in self.layout)     # bytes of a slot it takes
        self.ram: dict[str, torch.Tensor] | None = None     # the copy in the RAM cache
        self.spill: _Job | None = None                      # being written to its file
        self.stays_in_ram = False                           # the write failed or the drive is full: never written
        self.on_disk = False
        self.file_bytes = 0
        self.wanted = False                                 # asked for with want() and not yet taken
        self.skip_slot = False                              # bringing it into a slot failed: take() copies it itself
        self.slot: _Slot | None = None                      # the slot being filled for it or holding it
        self.device: dict[str, torch.Tensor] | None = None  # uploaded ahead, not yet taken
        self.upload: tuple[_Slot, Any] | None = None        # the slot and event of that upload
        self.touch = 0                                      # when it was last put, wanted or taken


class _Prop:
    __slots__ = ("id", "folder", "small", "sizes", "large", "failed")

    def __init__(self, prop_id: str, folder: Path):
        self.id = prop_id
        self.folder = folder
        self.small: dict[str, Any] = {}
        self.sizes: dict[str, int] = {}
        self.large: dict[str, _Item] = {}
        self.failed: tuple[str, str] | None = None          # (detail, kind) once the prop cannot go on


class _Job:
    """One piece of work for the helper thread, and what it reports back. `lent`, `buffer` and `views` are tensors
    of the GPU thread (`dst`: addresses inside a slot): the helper reads and fills their memory, the GPU thread
    alone lets go of them."""
    __slots__ = ("kind", "prop_id", "key", "path", "prop", "item", "slot", "lent", "dst", "names", "buffer",
                 "views", "args", "started", "cancelled", "finished", "error", "result")

    def __init__(self, kind: str, **fields: Any):
        for name in self.__slots__:
            setattr(self, name, None)
        self.kind = kind
        self.started = self.cancelled = self.finished = False
        for name, value in fields.items():
            setattr(self, name, value)


class PropStore:
    """See the top of this file. `ram_mib` / `slot_mib` None: what `ram_defaults()` says for this PC."""

    def __init__(self, work_dir: Path, device: str = "cuda", *, ram_mib: int | None = 768, slots: int = 2,
                 slot_mib: int | None = 256, on_event: Callable[[dict], None] | None = None):
        if ram_mib is None or slot_mib is None:
            default_ram, default_slot = ram_defaults()
            ram_mib = default_ram if ram_mib is None else ram_mib
            slot_mib = default_slot if slot_mib is None else slot_mib
        if slot_mib < 1 or slot_mib & (slot_mib - 1):
            raise ValueError(f"slot_mib must be a power of two, not {slot_mib}: page-locked memory comes in powers "
                             f"of two and the rest would be locked for nothing")
        if ram_mib < 0 or slots < 0:
            raise ValueError("ram_mib and slots cannot be negative")
        self._work_dir = Path(work_dir)
        self._device = torch.device(device)
        self._cuda = self._device.type == "cuda"
        self._card = self._device.index             # for torch.cuda's calls; None ("cuda"): the current device
        self._ram_limit = int(ram_mib) * MIB
        self._slot_count = int(slots)
        self._slot_bytes = int(slot_mib) * MIB
        self._on_event = on_event

        self._props: dict[str, _Prop] = {}
        self._cache: dict[_Item, None] = {}         # the items whose tensors are in RAM, oldest first
        self._wants: list[_Item] = []               # wanted, waiting for a free slot
        self._slots: list[_Slot] | None = None      # made at the first want()
        self._checkpoints: dict[str, int] = {}      # prop id -> bytes of its checkpoint on disk
        self._clock = 0
        self._total = 0                             # the card's memory, asked once

        self._cv = threading.Condition()            # guards _jobs, _done, _running, _stop
        self._jobs: collections.deque[_Job] = collections.deque()
        self._done: list[_Job] = []
        self._running: _Job | None = None
        self._thread: threading.Thread | None = None
        self._stop = False
        self._broken: str | None = None         # the helper thread stopped by itself: no file work will be done
        self._closed = False
        self._halt = threading.Event()              # cuts the helper's waits between two tries of a delete short

        self._ram_bytes = 0
        self._peak_ram_bytes = 0
        self._spilling_bytes = 0
        self._small_bytes = 0
        self._disk_bytes = 0
        self._device_bytes = 0
        self._pinned_bytes = 0
        self._spills = 0
        self._spilled_bytes = 0
        self._uploads_ahead = 0
        self._uploads_at_take = 0
        self._warnings = 0
        self._unpinned_noted = False

    # =================================================================================================================
    # The batch and its props
    # =================================================================================================================

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    def begin(self, keep: dict[str, str]) -> dict[str, dict]:
        """Start of a batch. Forgets every prop still open and empties the work folder of the store's files,
        except the checkpoints of `keep` = {prop_id: request_hash} whose hash, size and SHA-256 hold. Returns those:
        {prop_id: {"request_hash", "notes", "tensors": {name: CPU tensor}}} - the props to resume at their last
        part. A prop with an empty hash is never resumed."""
        for prop_id in list(self._props):
            self._forget(prop_id)
        self._forget_checkpoints()
        keep = {_plain(prop_id, "prop id"): str(request_hash) for prop_id, request_hash in keep.items()
                if request_hash}
        job = self._submit("begin", path=self._work_dir, args=keep)
        self._await(job, "the work folder to be emptied")
        if job.error is not None:
            self._warn("begin", f"{self._work_dir} could not be emptied: {job.error[1]}")
            return {}
        found, problems = job.result
        for problem in problems:
            self._warn("delete", problem)
        resumed: dict[str, dict] = {}
        for prop_id, record in found.items():
            folder = self._work_dir / prop_id
            buffer = torch.empty(record["need"], dtype=torch.uint8)
            read = self._submit("read", prop_id=prop_id, path=folder / f"{CHECKPOINT}.safetensors", buffer=buffer)
            self._await(read, f"the checkpoint of {prop_id}")
            views, read.result = read.result, None
            if read.error is not None:
                self._warn("checkpoint", f"the checkpoint of {prop_id} could not be read: {read.error[1]}", prop_id)
                self._submit("clear", prop_id=prop_id, path=folder)
                continue
            resumed[prop_id] = {"request_hash": keep[prop_id], "notes": record["notes"],
                                "tensors": {name: view.clone() for name, view in views.items()}}
            self._checkpoints[prop_id] = record["bytes"]
            self._disk_bytes += record["bytes"]
            del views, buffer
        return resumed

    def open_prop(self, prop_id: str) -> None:
        """A prop of the batch: its items can be put from now on, and its folder is made."""
        if prop_id in self._props:
            return
        folder = self._work_dir / _plain(prop_id, "prop id")
        self._props[prop_id] = _Prop(prop_id, folder)
        self._submit("mkdir", prop_id=prop_id, path=folder)

    def close_prop(self, prop_id: str) -> None:
        """Everything of the prop goes: its items now, its folder and checkpoint by the helper thread. Also for a
        prop that was never opened in this batch (a checkpoint `begin` kept)."""
        self._forget(prop_id)
        try:
            folder = self._work_dir / _plain(prop_id, "prop id")
        except ValueError:
            return
        self._disk_bytes -= self._checkpoints.pop(prop_id, 0)
        self._cancel_where(lambda job: job.prop_id == prop_id and job.kind in ("mkdir", "spill", "checkpoint",
                                                                               "delete"))
        self._submit("clear", prop_id=prop_id, path=folder)

    def flush(self) -> None:
        """Wait, pumping, until the helper thread has nothing left to do."""
        self._settle_queue(PATIENCE)

    def close(self) -> None:
        """End of the store: every prop is forgotten, queued checkpoints and deletes are finished, the helper
        thread is joined. Files of props that were still open stay for the next `begin` on the folder."""
        if self._closed:
            return
        for prop_id in list(self._props):
            self._forget(prop_id)
        self._cancel_where(lambda job: job.kind in ("mkdir", "spill", "fill", "read"))
        self._settle_queue(CLOSE_PATIENCE)
        self._closed = True
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._drain()
        if self._slots:
            for slot in self._slots:
                self._synchronise(slot)
            # Every buffer goes back to the spare pool - a page-locked block that is dropped is lost for the life
            # of the process. Only one the helper thread may still be filling stays behind, and only while that
            # thread is alive.
            busy = self._thread is not None and self._thread.is_alive()
            with _spare_lock:
                _spare.setdefault((self._slot_bytes, self._cuda), []).extend(
                    slot.buf for slot in self._slots if slot.state != FILLING or not busy)
        self._slots = None
        self._pinned_bytes = 0

    def stats(self) -> dict:
        """Bytes in the RAM cache, page-locked, on disk and uploaded ahead; counts of spills, uploads and warnings.
        `disk_bytes` is advisory: a spill whose item was discarded while it was being written is not counted,
        although its file is there until the helper thread has deleted it."""
        return {"ram_bytes": self._ram_bytes, "ram_limit": self._ram_limit, "peak_ram_bytes": self._peak_ram_bytes,
                "spilling_bytes": self._spilling_bytes, "small_bytes": self._small_bytes,
                "pinned_bytes": self._pinned_bytes,
                "slot_bytes": self._slot_bytes * len(self._slots) if self._slots else 0,
                "disk_bytes": self._disk_bytes, "device_bytes": self._device_bytes,
                "spills": self._spills, "spilled_bytes": self._spilled_bytes,
                "uploads_ahead": self._uploads_ahead, "uploads_at_take": self._uploads_at_take,
                "queued": len(self._jobs) + (self._running is not None), "props": len(self._props),
                "warnings": self._warnings}

    # =================================================================================================================
    # Small items: RAM only
    # =================================================================================================================

    def put_small(self, prop_id: str, key: str, value: Any) -> None:
        """A JSON-able object, a PIL picture or a dict of CPU tensors. The object itself is kept: do not change it."""
        prop = self._open(prop_id)
        self.discard(prop_id, key)
        prop.small[key] = value
        prop.sizes[key] = size = _small_size(value)
        self._small_bytes += size

    def get_small(self, prop_id: str, key: str) -> Any:
        prop = self._open(prop_id)
        if key not in prop.small:
            raise KeyError(f"prop {prop_id!r} has no small item {key!r}")
        return prop.small[key]

    # =================================================================================================================
    # Large items: card -> RAM cache (-> disk) -> slot -> card
    # =================================================================================================================

    def put_device(self, prop_id: str, key: str, tensors: dict[str, torch.Tensor]) -> None:
        """Tensors that live on the card: a blocking copy into the RAM cache. When the cache is over its budget
        the items needed longest ago go to disk; while such writes still hold the cache over the budget this call
        waits for them first, so the cache never grows by more than the item put."""
        prop = self._open(prop_id)
        if _plain(key, "key of a large item") == CHECKPOINT:
            raise ValueError(f"{CHECKPOINT!r} is the checkpoint's name")
        self.discard(prop_id, key)
        self._make_room()
        resting = {name: _resting(tensor) for name, tensor in tensors.items()}
        item = _Item(prop, key, resting)
        item.touch = self._tick_clock()
        prop.large[key] = item
        self._set_ram(item, resting)
        self._evict()

    def want(self, prop_id: str, key: str) -> None:
        """Start bringing a large item into a slot, so that `pump()` can upload it before its prop's turn. Nothing
        happens for an item that is larger than a slot or unknown: `take()` then copies it itself."""
        prop = self._props.get(prop_id)
        item = prop.large.get(key) if prop is not None else None
        if item is None:
            return
        self._tick()
        if self._slots is None:
            self._make_slots()
        item.touch = self._tick_clock()
        if (item.wanted or item.skip_slot or item.device is not None or item.slot is not None
                or not self._slots or item.need > self._slot_bytes):
            return
        item.wanted = True
        self._wants.append(item)
        self._assign()

    def pump(self, upload: bool = True) -> None:
        """Called from every sampler step and at prop boundaries; never waits. Takes the helper thread's reports
        (drops the RAM copy of an item that reached its file), frees the slots whose copies have run, gives free
        slots to wanted items, and uploads staged items while the card has room for them under its cap.
        `upload=False` (a prop that is short of memory): nothing is uploaded ahead, and what was is given back."""
        self._tick()
        if not self._slots:
            return
        if upload:
            self._upload_ahead()
        else:
            self._hold_back()

    def take(self, prop_id: str, key: str) -> dict[str, torch.Tensor]:
        """The item's tensors on the card, owned by the caller. At once when `pump()` uploaded them; else from the
        slot, from the RAM cache or from the file, waiting (and pumping) as long as that takes. The item itself
        stays in the store until it is discarded."""
        prop = self._open(prop_id)
        item = prop.large.get(key)
        if item is None:
            raise KeyError(f"prop {prop_id!r} has no large item {key!r}")
        self._tick()
        item.touch = self._tick_clock()
        self._unwant(item)
        if item.device is not None:
            tensors, item.device, item.upload = item.device, None, None
            self._device_bytes -= item.nbytes
            return tensors
        slot = item.slot
        if slot is not None and slot.state == FILLING:
            job = slot.job
            if item.ram is None or not self._cancel(job):
                self._await(job, f"{key} of {prop_id} to be staged")
            slot = item.slot
        if slot is not None and slot.state == STAGED:
            self._uploads_at_take += 1
            return self._issue(slot)
        self._uploads_at_take += 1
        if item.ram is not None:
            return {name: self._copy(tensor) for name, tensor in item.ram.items()}
        return self._from_file(item)

    def discard(self, prop_id: str, *keys: str) -> None:
        """Items the prop needs no more: out of RAM now, the file by the helper thread."""
        prop = self._props.get(prop_id)
        if prop is None:
            return
        for key in keys:
            if key in prop.small:
                del prop.small[key]
                self._small_bytes -= prop.sizes.pop(key)
            item = prop.large.pop(key, None)
            if item is not None:
                self._drop(item, delete=True)

    # =================================================================================================================
    # The checkpoint
    # =================================================================================================================

    def checkpoint(self, prop_id: str, request_hash: str, tensors: dict[str, torch.Tensor], notes: dict) -> None:
        """The prop's final latents to disk, so that a restart resumes it at its last part: final.safetensors
        (flushed to the disk), then final.json with the hash of the request, the notes and the file's size and
        SHA-256. Written by the helper thread; a failure only warns (the resume is lost, not the prop)."""
        prop = self._open(prop_id)
        if not request_hash:
            return
        try:
            notes_text = json.dumps(notes)
            resting = {name: _resting(tensor) for name, tensor in tensors.items()}
        except (TypeError, ValueError, AttributeError) as error:
            self._warn("checkpoint", f"the checkpoint of {prop_id} was not written (its notes must be JSON, its "
                                     f"latents tensors): {error}", prop_id)
            return
        self._cancel_where(lambda job: job.kind == "checkpoint" and job.prop_id == prop_id)
        self._submit("checkpoint", prop_id=prop_id, prop=prop, path=prop.folder / f"{CHECKPOINT}.safetensors",
                     lent=resting, args=(str(request_hash), notes_text))

    # =================================================================================================================
    # GPU thread: items
    # =================================================================================================================

    def _open(self, prop_id: str) -> _Prop:
        prop = self._props.get(prop_id)
        if prop is None:
            raise KeyError(f"prop {prop_id!r} is not open in the store")
        if prop.failed is not None:
            raise PropFailed(*prop.failed)
        return prop

    def _tick_clock(self) -> int:
        self._clock += 1
        return self._clock

    def _set_ram(self, item: _Item, tensors: dict[str, torch.Tensor] | None) -> None:
        """Put the item's tensors into the RAM cache, or (None) let go of them."""
        if item.ram is not None:
            self._ram_bytes -= item.nbytes
            self._cache.pop(item, None)
        item.ram = tensors
        if tensors is not None:
            self._ram_bytes += item.nbytes
            self._peak_ram_bytes = max(self._peak_ram_bytes, self._ram_bytes)
            self._cache[item] = None

    def _unwant(self, item: _Item) -> None:
        if item.wanted:
            item.wanted = False
            if item in self._wants:
                self._wants.remove(item)

    def _forget_checkpoints(self) -> None:
        self._disk_bytes -= sum(self._checkpoints.values())
        self._checkpoints.clear()

    def _forget(self, prop_id: str) -> None:
        """Drop everything the store holds of a prop in memory. No file work."""
        prop = self._props.pop(prop_id, None)
        if prop is None:
            return
        for item in list(prop.large.values()):
            self._drop(item, delete=False)
        prop.large.clear()
        prop.small.clear()
        self._small_bytes -= sum(prop.sizes.values())
        prop.sizes.clear()

    def _drop(self, item: _Item, delete: bool) -> None:
        """Let go of a large item: its upload is waited for, its slot handed back (at once, or when the helper
        thread has finished with it), its RAM copy dropped, its file deleted by the helper thread."""
        self._unwant(item)
        if item.device is not None:
            self._upload_over(item, wait=True)
            item.device = None
            self._device_bytes -= item.nbytes
        slot = item.slot
        if slot is not None:
            item.slot, slot.item, slot.views = None, None, None
            if slot.state == STAGED:
                slot.state = FREE
            elif slot.state == FILLING:
                self._cancel(slot.job)
        written = item.on_disk
        if item.spill is not None:
            job, item.spill = item.spill, None
            self._spilling_bytes -= item.nbytes
            written = not self._cancel(job) or written
        if item.ram is not None:
            self._set_ram(item, None)
        if item.on_disk:
            item.on_disk = False
            self._disk_bytes -= item.file_bytes
        if delete and written:
            self._submit("delete", prop_id=item.prop.id, key=item.key, path=item.path)

    def _fail(self, prop: _Prop, detail: str, kind: str) -> None:
        """The prop cannot go on: its large items are dropped now, and its next take / get / put raises."""
        if prop.failed is None:
            prop.failed = (detail, kind)
        for item in list(prop.large.values()):
            self._drop(item, delete=True)
        prop.large.clear()

    def _make_room(self) -> None:
        """Wait while writes that are under way still hold the RAM cache over its budget."""
        deadline = time.monotonic() + PATIENCE
        while (self._spilling_bytes and self._ram_bytes > self._ram_limit and self._broken is None
               and time.monotonic() < deadline):
            self._tick()
            if self._spilling_bytes and self._ram_bytes > self._ram_limit:
                self._nap()

    def _evict(self) -> None:
        """Over the budget: hand the items needed longest ago to the helper thread, which writes them to their
        files. Not the ones a want() names, and not the ones whose write already failed."""
        over = self._ram_bytes - self._spilling_bytes - self._ram_limit
        if over <= 0:
            return
        for item in sorted(self._cache, key=lambda entry: entry.touch):
            if item.spill is not None or item.stays_in_ram or item.wanted or item.slot is not None:
                continue
            item.spill = self._submit("spill", prop_id=item.prop.id, key=item.key, path=item.path, item=item,
                                      lent=item.ram, args=item.nbytes)
            self._spilling_bytes += item.nbytes
            over -= item.nbytes
            if over <= 0:
                return

    # =================================================================================================================
    # GPU thread: slots and uploads
    # =================================================================================================================

    def _make_slots(self) -> None:
        """The slots, once: the first want() of the process allocates them (a CUDA call for a card)."""
        self._slots = []
        for _ in range(self._slot_count):
            buf = _borrow(self._slot_bytes, self._cuda)
            self._slots.append(_Slot(buf, self._cuda and buf.is_pinned()))
        self._pinned_bytes = sum(slot.buf.numel() for slot in self._slots if slot.pinned)
        if self._cuda and any(not slot.pinned for slot in self._slots):
            self._warn("pin", "Windows did not lock the store's upload buffers in memory: uploads wait for their "
                              "copies (same values, milliseconds slower)")

    def _views(self, slot: _Slot, item: _Item) -> dict[str, torch.Tensor]:
        """Where the item's tensors lie in a slot: views of the slot's own tensor (torch sees them as page-locked
        only then), each starting on a multiple of ALIGN."""
        views: dict[str, torch.Tensor] = {}
        at = 0
        for name, dtype, shape, size in item.layout:
            views[name] = slot.buf[at:at + size].view(dtype).view(shape)
            at += _aligned(size)
        return views

    def _lend(self, slot: _Slot, item: _Item) -> _Job:
        """Give a free slot to the helper thread to fill with the item: from its RAM copy, else from its file."""
        if slot.state != FREE or slot.event is not None:
            raise RuntimeError("a slot was about to be refilled before its copies had run")
        slot.state, slot.item, item.slot = FILLING, item, slot
        if item.ram is not None:
            views = self._views(slot, item)
            moves = [(views[name].data_ptr(), item.ram[name].data_ptr(), size) for name, _, _, size in item.layout]
            slot.job = self._submit("fill", prop_id=item.prop.id, key=item.key, item=item, slot=slot,
                                    lent=item.ram, dst=moves, views=views)
        else:
            slot.job = self._submit("fill", prop_id=item.prop.id, key=item.key, item=item, slot=slot,
                                    path=item.path, names=item.names, buffer=slot.buf)
        return slot.job

    def _assign(self) -> None:
        """Free slots to the wanted items that wait for one, first asked first."""
        while self._wants:
            slot = next((slot for slot in self._slots if slot.state == FREE), None)
            if slot is None:
                return
            self._lend(slot, self._wants.pop(0))

    def _release(self, slot: _Slot) -> None:
        slot.event, slot.state = None, FREE

    def _synchronise(self, slot: _Slot) -> None:
        """Wait for the copies out of a slot, then it is free."""
        if slot.state == UPLOADING:
            slot.event.synchronize()
            self._release(slot)

    def _copy(self, tensor: torch.Tensor) -> torch.Tensor:
        """A CPU tensor onto the store's device, waiting for the copy. Always new memory, on the CPU too."""
        return tensor.to(self._device) if self._cuda else tensor.clone()

    def _issue(self, slot: _Slot) -> dict[str, torch.Tensor]:
        """Copy a staged item from its slot to the card and part the two. From page-locked memory the copies are
        only queued (behind the kernels already on the stream) and an event behind the last one says when the slot
        may be used again; otherwise each copy has run when this returns."""
        item, views, direct = slot.item, slot.views, slot.direct
        tensors: dict[str, torch.Tensor] = {}
        try:
            for name, view in views.items():
                tensors[name] = view.to(self._device, non_blocking=True) if direct else self._copy(view)
        finally:
            item.slot, slot.item, slot.views = None, None, None
            if direct:
                slot.event = torch.cuda.Event()
                slot.event.record(torch.cuda.current_stream(self._card))
                slot.state = UPLOADING
            else:
                slot.state = FREE
        return tensors

    def _fits(self, nbytes: int) -> bool:
        """Has the card room for an upload ahead of its turn: what PyTorch has reserved, the item and the headroom
        stay under PyTorch's cap; with no cap set, the item and the headroom fit into the card's free memory."""
        if not self._cuda:
            return True
        fraction = torch.cuda.get_per_process_memory_fraction(self._card)
        if fraction < 1.0:
            if not self._total:
                self._total = torch.cuda.get_device_properties(self._card).total_memory
            return torch.cuda.memory_reserved(self._card) + nbytes + UPLOAD_HEADROOM <= int(fraction * self._total)
        return nbytes + UPLOAD_HEADROOM <= torch.cuda.mem_get_info(self._card)[0]

    def _upload_ahead(self) -> None:
        for slot in self._slots:
            item = slot.item
            if slot.state != STAGED or item is None or not item.wanted or not self._fits(item.nbytes):
                continue
            refused = None
            try:
                tensors = self._issue(slot)
            except torch.OutOfMemoryError as error:
                refused = str(error).splitlines()[0]
            if refused is not None:
                self._unwant(item)
                item.skip_slot = True
                self._warn("upload", f"{item.key} of {item.prop.id} was not uploaded ahead: {refused}",
                           item.prop.id, item.key)
                continue
            item.device = tensors
            item.upload = (slot, slot.event) if slot.event is not None else None
            self._device_bytes += item.nbytes
            self._uploads_ahead += 1

    def _upload_over(self, item: _Item, wait: bool) -> bool:
        """Have the copies of an item that was uploaded ahead run? Its slot is then free again. With `wait` they
        are waited for."""
        if item.upload is not None:
            slot, event = item.upload
            if slot.event is event:
                if wait:
                    self._synchronise(slot)
                elif event.query():
                    self._release(slot)
                else:
                    return False
            item.upload = None
        return True

    def _hold_back(self) -> None:
        """Give the card back what was uploaded ahead and not taken, once its copies have run; the items are
        staged again and take() uploads them at their prop's turn."""
        if not self._device_bytes:
            return
        for prop in self._props.values():
            for item in prop.large.values():
                if item.device is None or not self._upload_over(item, wait=False):
                    continue
                item.device = None
                self._device_bytes -= item.nbytes
                if item.wanted:
                    self._wants.append(item)
        self._assign()

    def _from_file(self, item: _Item) -> dict[str, torch.Tensor]:
        """A spilled item to the card: through a slot when one is free or can be waited free, else through a
        pageable buffer of this thread's (three demands on two slots do not wait for each other)."""
        if not item.on_disk:
            raise RuntimeError(f"{item.key} of {item.prop.id} is neither in RAM nor on disk")
        slot = None
        if self._slots and not item.skip_slot and item.need <= self._slot_bytes:
            slot = next((slot for slot in self._slots if slot.state == FREE), None)
            if slot is None:
                slot = next((slot for slot in self._slots if slot.state == UPLOADING), None)
                if slot is not None:
                    self._synchronise(slot)
        if slot is not None:
            job = self._lend(slot, item)
            self._await(job, f"{item.key} of {item.prop.id} to be read")
            if job.error is not None:
                raise _as_exception(job.error)
            if item.slot is None:               # another report let the prop fail while this read ran, and the
                self._open(item.prop.id)        # item went with it: say why, in the words the artist is shown
                raise RuntimeError(f"{item.key} of {item.prop.id} was dropped while it was being read")
            return self._issue(item.slot)
        buffer = torch.empty(item.need, dtype=torch.uint8)
        job = self._submit("read", prop_id=item.prop.id, key=item.key, path=item.path,
                           names=item.names, buffer=buffer)
        self._await(job, f"{item.key} of {item.prop.id} to be read")
        views, job.result = job.result, None
        if job.error is not None:
            raise _as_exception(job.error)
        problem = _mismatch(item, views)
        if problem:
            raise RuntimeError(f"{item.path}: {problem}")
        return {name: self._copy(views[name]) for name in item.names}

    # =================================================================================================================
    # GPU thread: the helper thread's queue and its reports
    # =================================================================================================================

    def _submit(self, kind: str, **fields: Any) -> _Job:
        if self._closed:
            raise RuntimeError("the store is closed")
        job = _Job(kind, **fields)
        with self._cv:
            if self._thread is None:
                self._thread = threading.Thread(target=self._helper, name="p3d-store-io", daemon=True)
                self._thread.start()
            self._jobs.append(job)
            self._cv.notify_all()
        return job

    def _cancel(self, job: _Job) -> bool:
        """Take a job back that the helper thread has not begun. False when it has: the job then runs to its end
        and its report is handled as any other (by then the item it was for may be gone: the handlers look)."""
        with self._cv:
            if job.started or job.finished:
                return False
            job.cancelled = True
            self._jobs.remove(job)
        self._finish(job)
        return True

    def _cancel_where(self, match: Callable[[_Job], bool]) -> None:
        with self._cv:
            taken = [job for job in self._jobs if match(job)]
            for job in taken:
                job.cancelled = True
                self._jobs.remove(job)
        for job in taken:
            self._finish(job)

    def _nap(self) -> None:
        with self._cv:
            if not self._done:
                self._cv.wait(WAIT)

    def _await(self, job: _Job, what: str) -> None:
        """Wait, pumping, until the helper thread has done this job and its report was handled."""
        deadline = time.monotonic() + PATIENCE
        while not job.finished:
            self._tick()
            if job.finished:
                return
            if self._broken is not None:
                raise RuntimeError(f"Pixal3D's temporary files: the thread that reads and writes them stopped "
                                   f"({self._broken}), waiting for {what}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Pixal3D's temporary files: waited {PATIENCE:.0f} s for {what}")
            self._nap()

    def _settle_queue(self, patience: float) -> None:
        deadline = time.monotonic() + patience
        while True:
            self._tick()
            with self._cv:
                if not self._jobs and self._running is None and not self._done:
                    return
                if self._broken is not None or time.monotonic() > deadline:
                    break
                if not self._done:
                    self._cv.wait(WAIT)
        if self._broken is not None:
            self._warn("flush", f"the store's file work was left undone: the thread that does it stopped "
                                f"({self._broken})")
        else:
            self._warn("flush", f"the store's file work was not done after {patience:.0f} s")

    def _tick(self) -> None:
        """One round of housekeeping, without waiting: reports, finished copies, free slots, the budget."""
        if self._done:
            self._drain()
        if self._slots:
            for slot in self._slots:
                if slot.state == UPLOADING and slot.event.query():
                    self._release(slot)
            if self._wants:
                self._assign()
        if self._ram_bytes - self._spilling_bytes > self._ram_limit:
            self._evict()

    def _drain(self) -> None:
        with self._cv:
            done, self._done = self._done, []
        for job in done:
            self._finish(job)

    def _finish(self, job: _Job) -> None:
        """Handle one report of the helper thread (or a job taken back before it ran): the one place where the
        GPU thread lets go of what the job was lent."""
        job.lent, job.dst, job.buffer, job.views = None, None, None, None
        if job.kind == "spill":
            self._spilled(job)
        elif job.kind == "fill":
            self._filled(job)
        elif job.kind == "checkpoint":
            if not job.cancelled and job.error is not None:
                self._warn("checkpoint", f"the checkpoint of {job.prop_id} was not written (a restart cannot resume "
                                         f"this prop): {job.error[1]}", job.prop_id)
            elif not job.cancelled and self._props.get(job.prop_id) is job.prop:
                self._disk_bytes += job.result - self._checkpoints.get(job.prop_id, 0)
                self._checkpoints[job.prop_id] = job.result
        elif job.kind in ("delete", "clear", "mkdir"):
            if not job.cancelled and job.error is not None:
                self._warn("delete" if job.kind != "mkdir" else "folder", job.error[1], job.prop_id, job.key)
        job.finished = True

    def _spilled(self, job: _Job) -> None:
        item = job.item
        if item.spill is not job:
            return
        item.spill = None
        self._spilling_bytes -= item.nbytes
        if job.cancelled:
            return
        if job.error is not None:
            item.stays_in_ram = True
            self._warn("spill", f"{item.key} of {item.prop.id} stays in RAM: {job.error[1]}", item.prop.id, item.key)
            if _ram_available() < RAM_FLOOR:
                self._fail(item.prop, DISK_MESSAGE, "disk")
            return
        item.on_disk, item.file_bytes = True, job.result
        self._disk_bytes += item.file_bytes
        self._spills += 1
        self._spilled_bytes += item.nbytes
        self._set_ram(item, None)

    def _filled(self, job: _Job) -> None:
        slot, item = job.slot, job.item
        views, job.result = job.result, None
        slot.job = None
        if slot.item is not item:
            slot.state = FREE
            return
        if job.cancelled:
            slot.state, slot.item, item.slot = FREE, None, None
            return
        problem = job.error[1] if job.error is not None else _mismatch(item, views)
        if problem:
            slot.state, slot.item, item.slot = FREE, None, None
            item.skip_slot = True
            self._unwant(item)
            if job.error is None:
                job.error = ("RuntimeError", f"{item.path}: {problem}")
            self._warn("stage", f"{item.key} of {item.prop.id} could not be staged: {problem}", item.prop.id,
                       item.key)
            return
        slot.views, slot.state = {name: views[name] for name in item.names}, STAGED
        slot.direct = slot.pinned and all(view.is_pinned() for view in views.values() if view.numel())
        if slot.pinned and not slot.direct and not self._unpinned_noted:
            self._unpinned_noted = True
            self._warn("pin", "views of a page-locked slot are not seen as page-locked: uploads wait for their "
                              "copies")

    def _warn(self, what: str, message: str, prop_id: str | None = None, key: str | None = None) -> None:
        """A `store_warning` event. Only ever sent from the GPU thread."""
        self._warnings += 1
        if self._on_event is None:
            print(f"[store] {what}: {message}", flush=True)
            return
        try:
            self._on_event({"kind": "store_warning", "what": what, "prop_id": prop_id, "key": key,
                            "message": message})
        except Exception as error:
            print(f"[store] on_event raised {type(error).__name__}: {error} (for: {what}: {message})", flush=True)

    # =================================================================================================================
    # Helper thread: file work and filling slots. No tensor is made or freed here, and CUDA is never called.
    # =================================================================================================================

    def _helper(self) -> None:
        try:
            while True:
                with self._cv:
                    while not self._jobs and not self._stop:
                        self._cv.wait()
                    if not self._jobs:
                        return
                    job = self._jobs.popleft()
                    job.started = True
                    self._running = job
                self._work(job)
                with self._cv:                  # the report is read under this lock: by then this thread holds
                    self._running = None        # nothing of the job any more
                    self._done.append(job)
                    del job
                    self._cv.notify_all()
        except BaseException as error:          # the work itself cannot raise (_work catches it), the queue around
            with self._cv:                      # it can; nobody restarts this thread, so the waits of the GPU
                self._broken = f"{type(error).__name__}: {error}"    # thread give up at once from now on
                self._running = None
                self._cv.notify_all()

    def _work(self, job: _Job) -> None:
        try:
            job.result = getattr(self, f"_do_{job.kind}")(job)
        except BaseException as error:
            job.result = None
            job.error = (type(error).__name__,
                         str(error) if isinstance(error, _Refused) else f"{type(error).__name__}: {error}")

    def _do_mkdir(self, job: _Job) -> None:
        os.makedirs(job.path, exist_ok=True)

    def _do_spill(self, job: _Job) -> int:
        """Write an item of the RAM cache to its file, straight from the tensors' memory."""
        folder = job.path.parent
        os.makedirs(folder, exist_ok=True)
        free = _free_bytes(folder)
        if free - job.args < DISK_FLOOR:
            raise _Refused(f"only {free // MIB} MiB are free on the drive of {folder}")
        return stfile.write(job.path, job.lent)

    def _do_read(self, job: _Job) -> dict[str, torch.Tensor]:
        """A file into a buffer of the GPU thread's; the report says where in it the item's tensors lie."""
        return stfile.read_into(job.path, job.names, job.buffer)

    def _do_fill(self, job: _Job) -> dict[str, torch.Tensor] | None:
        """Bring an item into a slot: its file read into the slot, or its bytes copied from the RAM cache to where
        the GPU thread laid its tensors out in the slot (plain memory copies: a torch copy would start a team of
        worker threads for this thread). The report is where the tensors lie, as for a read."""
        if job.lent is None:
            return self._do_read(job)
        for target, source, size in job.dst:
            if size:
                ctypes.memmove(target, source, size)
        return job.views

    def _do_checkpoint(self, job: _Job) -> int:
        request_hash, notes_text = job.args
        folder = job.path.parent
        record = folder / f"{CHECKPOINT}.json"
        os.makedirs(folder, exist_ok=True)
        problem = self._retry(os.remove, record)        # an older record must not describe the new file
        if problem:
            raise _Refused(problem)
        stfile.write(job.path, job.lent, fsync=True)
        size = os.path.getsize(job.path)
        text = json.dumps({"request_hash": request_hash, "notes": json.loads(notes_text), "bytes": size,
                           "sha256": _sha256(job.path)})
        partial = folder / f"{CHECKPOINT}.json.tmp"
        with open(partial, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, record)
        return size + len(text.encode("utf-8"))

    def _do_delete(self, job: _Job) -> None:
        problem = self._retry(os.remove, job.path)
        if problem:
            raise _Refused(problem)

    def _do_clear(self, job: _Job) -> None:
        problems = self._clear(job.path)
        if problems:
            raise _Refused("; ".join(problems))

    def _do_begin(self, job: _Job) -> tuple[dict[str, dict], list[str]]:
        """Empty the work folder of the store's files, but for the valid checkpoints of the props to keep."""
        root, keep = job.path, job.args
        os.makedirs(root, exist_ok=True)
        found: dict[str, dict] = {}
        problems: list[str] = []
        with os.scandir(root) as entries:
            listing = [(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries]
        for name, is_folder in sorted(listing):
            path = root / name
            if not is_folder:
                if name.endswith(".tmp"):
                    problem = self._retry(os.remove, path)
                    if problem:
                        problems.append(problem)
                continue
            record = self._verified(path, keep[name]) if name in keep else None
            if record is None:
                problems += self._clear(path)
                continue
            problems += self._clear(path, keep=(f"{CHECKPOINT}.safetensors", f"{CHECKPOINT}.json"))
            found[name] = record
        return found, problems

    def _verified(self, folder: Path, request_hash: str) -> dict | None:
        """The checkpoint in a prop's folder when it is the one of this request and whole: its notes, its size and
        the bytes a buffer needs to hold its tensors."""
        data, text = folder / f"{CHECKPOINT}.safetensors", folder / f"{CHECKPOINT}.json"
        try:
            record = json.loads(text.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("request_hash") != request_hash:
                return None
            size = os.path.getsize(data)
            if record.get("bytes") != size or record.get("sha256") != _sha256(data):
                return None
            need = sum(_aligned(info.end - info.start) for info in stfile.read_header(data).values())
            notes = record.get("notes")
            return {"notes": notes if isinstance(notes, dict) else {}, "bytes": size + os.path.getsize(text),
                    "need": need}
        except Exception:                               # whatever is wrong with it: it is not a checkpoint to trust
            return None

    def _clear(self, folder: Path, keep: tuple[str, ...] = ()) -> list[str]:
        """Delete the store's files in a prop's folder (but `keep`), then the folder when nothing is left in it.
        A file of another name is not the store's: it and the folder are left, and that is reported."""
        problems: list[str] = []
        foreign: list[str] = []
        kept = 0
        try:
            with os.scandir(folder) as entries:
                listing = [(entry.name, entry.is_file(follow_symlinks=False)) for entry in entries]
        except FileNotFoundError:
            return problems
        except OSError as error:
            return [f"{folder}: {type(error).__name__}: {error}"]
        for name, is_file in sorted(listing):
            if name in keep:
                kept += 1
            elif not is_file or not _ours(name):
                foreign.append(name)
            else:
                problem = self._retry(os.remove, folder / name)
                if problem:
                    problems.append(problem)
        if foreign:
            problems.append(f"{folder} holds {len(foreign)} entries that are not the store's ({foreign[0]}): they "
                            f"were left alone")
        elif not kept and not problems:
            problem = self._retry(os.rmdir, folder)
            if problem:
                problems.append(problem)
        return problems

    def _retry(self, remove: Callable[[Path], None], path: Path) -> str | None:
        """remove(path); when Windows will not let the file go (a virus scanner, the indexer, a sync client holds
        it open) three more tries over two seconds. What went wrong at last, or None; a missing file is fine."""
        problem = None
        for wait in (0.0, *DELETE_WAITS):
            if wait:
                self._halt.wait(wait)
            try:
                remove(path)
                return None
            except FileNotFoundError:
                return None
            except OSError as error:
                problem = (f"{path} could not be deleted after {len(DELETE_WAITS) + 1} tries: "
                           f"{error.strerror or type(error).__name__}")
        return problem


def _small_size(value: Any) -> int:
    """Roughly the bytes of a small item (for stats): a dict's tensors, a picture's pixels."""
    if isinstance(value, dict):
        return sum(entry.numel() * entry.element_size() for entry in value.values()
                   if isinstance(entry, torch.Tensor))
    size, bands = getattr(value, "size", None), getattr(value, "getbands", None)
    if callable(bands) and isinstance(size, tuple) and len(size) == 2:
        return size[0] * size[1] * len(bands())
    return 0


def _mismatch(item: _Item, views: dict[str, torch.Tensor]) -> str | None:
    """What is wrong with the tensors read back from an item's file, or None: they must be the ones put."""
    if set(views) != set(item.names):
        return f"the file holds {sorted(views)}, not the item's tensors"
    for name, dtype, shape, _ in item.layout:
        if views[name].dtype != dtype or tuple(views[name].shape) != shape:
            return f"{name} is {views[name].dtype} {list(views[name].shape)} in the file, not {dtype} {list(shape)}"
    return None


def _as_exception(error: tuple[str, str]) -> Exception:
    """The helper thread's error for the GPU thread to raise."""
    kind, text = error
    return FileNotFoundError(text) if kind == "FileNotFoundError" else RuntimeError(text)
