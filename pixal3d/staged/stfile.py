"""Safetensors files read and written with plain sequential file calls, never through a memory mapping (a mapped
file is read page fault by page fault: 0.1-0.75 GB/s cold on the PC this was measured on, against 1.5-2.3 GB/s
for sequential reads, and it charges the whole file to the process's commit while it is open).

Reading goes into ONE buffer the caller owns: a 1-D uint8 CPU tensor, pinned or not. Every tensor handed out is a
slice of that buffer seen in the tensor's own dtype and shape, so torch still knows it as pinned memory and a
non-blocking copy to the card stays non-blocking. Nothing here touches CUDA or allocates tensor memory.

    header = read_header(path)                          # name -> TensorInfo, in file order
    for name, info, data in iter_tensors(path, buffer): # one tensor at a time through the buffer (network loader)
        ...
    tensors = read_into(path, None, buffer)             # a whole small file into the buffer at once (store slots)
    write(path, {"feats": feats, "coords": coords})     # CPU tensors, through <path>.tmp

The format: 8 bytes (little-endian count N), N bytes of JSON {name: {"dtype", "shape", "data_offsets": [a, b]}},
then the tensors' bytes, each at a + 8 + N. Complex64 is stored as pairs of float32 (real, imaginary).
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

import torch

DTYPES = {"F32", "F16", "BF16", "I32", "I64", "U8", "BOOL", "C64"}

_TORCH = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I32": torch.int32,
          "I64": torch.int64, "U8": torch.uint8, "BOOL": torch.bool, "C64": torch.complex64}
_CODE = {dtype: code for code, dtype in _TORCH.items()}
_ITEM = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I64": 8, "U8": 1, "BOOL": 1, "C64": 8}
_HEADER_LIMIT = 100 << 20


class TooLarge(ValueError):
    """The tensors asked for do not fit the buffer given."""


class TensorInfo(NamedTuple):
    """Where one tensor lies in a file."""
    dtype: str                  # one of DTYPES
    shape: tuple[int, ...]
    start: int                  # its first byte, counted from the start of the file
    end: int                    # one past its last byte

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def itemsize(self) -> int:
        return _ITEM[self.dtype]

    @property
    def torch_dtype(self) -> torch.dtype:
        return _TORCH[self.dtype]


def read_header(path: str | os.PathLike) -> dict[str, TensorInfo]:
    """name -> TensorInfo of every tensor in the file, sorted by `start`. Reads the header only."""
    with open(path, "rb", buffering=0) as handle:
        return _header(handle, path)


def as_tensor(data: torch.Tensor, info: TensorInfo) -> torch.Tensor:
    """A tensor's bytes (1-D uint8, `info.nbytes` long, on any device) seen as the tensor itself: a view, no copy.
    The bytes must start at a multiple of the element size inside their storage."""
    if info.dtype == "C64":
        return torch.view_as_complex(data.view(torch.float32).view(*info.shape, 2))
    return data.view(_TORCH[info.dtype]).view(info.shape)


def iter_tensors(path: str | os.PathLike, buffer: torch.Tensor, names: Iterable[str] | None = None,
                 ) -> Iterator[tuple[str, TensorInfo, torch.Tensor | Iterator[torch.Tensor]]]:
    """The file's tensors one after another in file order (only `names` when given), each read into `buffer`.

    Yields (name, info, data). `data` is a view of `buffer` in the tensor's dtype and shape. For a tensor larger
    than the buffer it is instead an iterator of 1-D uint8 views of `buffer`: the tensor's bytes in order, piece
    by piece, every piece a whole number of elements (`as_tensor` turns the collected bytes into the tensor).
    Whatever was yielded is valid only until the next piece or tensor is asked for: the buffer is reused."""
    raw = _bytes_of(buffer)
    with open(path, "rb", buffering=0) as handle:
        for name, info in _select(_header(handle, path), names, path):
            handle.seek(info.start)
            if info.nbytes <= len(raw):
                _fill(handle, raw[:info.nbytes], path)
                yield name, info, as_tensor(buffer[:info.nbytes], info)
            else:
                pieces = _pieces(handle, info, buffer, raw, path)
                yield name, info, pieces
                pieces.close()


def read_into(path: str | os.PathLike, names: Iterable[str] | None, buffer: torch.Tensor) -> dict[str, torch.Tensor]:
    """The file's tensors (only `names` when given) all at once in `buffer`: name -> a view of the buffer in the
    tensor's dtype and shape, valid until the buffer is used for something else. Raises TooLarge when they do
    not fit; the buffer is then untouched."""
    raw = _bytes_of(buffer)
    with open(path, "rb", buffering=0) as handle:
        places: list[tuple[str, TensorInfo, int]] = []
        need = 0
        for name, info in _select(_header(handle, path), names, path):
            need += -need % info.itemsize               # a dtype view must start on a multiple of its element size
            places.append((name, info, need))
            need += info.nbytes
        if need > len(raw):
            raise TooLarge(f"{path}: {need} bytes asked for, the buffer holds {len(raw)}")
        tensors: dict[str, torch.Tensor] = {}
        for name, info, at in places:
            handle.seek(info.start)
            _fill(handle, raw[at:at + info.nbytes], path)
            tensors[name] = as_tensor(buffer[at:at + info.nbytes], info)
    return tensors


def write(path: str | os.PathLike, tensors: dict[str, torch.Tensor], *, fsync: bool = False) -> int:
    """Write CPU tensors as one safetensors file and return its size in bytes. The file is written as <path>.tmp
    and renamed over `path`, so a reader never sees half of it; with `fsync` it is on the disk before the rename.
    The bytes go out straight from each tensor's memory (a tensor that is not contiguous is copied first)."""
    entries: list[tuple[str, str, torch.Tensor, int]] = []
    for name, tensor in tensors.items():
        if not isinstance(name, str) or name == "__metadata__":
            raise ValueError(f"{name!r} cannot be a tensor's name")
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError(f"{name}: only dense CPU tensors can be written")
        code = _CODE.get(tensor.dtype)
        if code is None:
            raise ValueError(f"{name}: {tensor.dtype} cannot be written (supported: {sorted(DTYPES)})")
        if tensor.is_conj() or tensor.is_neg():
            tensor = tensor.resolve_conj().resolve_neg()
        tensor = tensor.contiguous()
        entries.append((name, code, tensor, tensor.numel() * _ITEM[code]))
    entries.sort(key=lambda entry: (-_ITEM[entry[1]], entry[0]))    # largest elements first: every tensor aligned

    header: dict[str, dict] = {}
    total = 0
    for name, code, tensor, size in entries:
        header[name] = {"dtype": code, "shape": list(tensor.shape), "data_offsets": [total, total + size]}
        total += size
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)

    partial = f"{os.fspath(path)}.tmp"
    try:
        with open(partial, "wb", buffering=0) as handle:
            _drain(handle, memoryview(struct.pack("<Q", len(text)) + text))
            for name, code, tensor, size in entries:
                if size:
                    _drain(handle, memoryview((ctypes.c_ubyte * size).from_address(tensor.data_ptr())))
            if fsync:
                os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        try:
            os.unlink(partial)
        except OSError:
            pass
        raise
    return 8 + len(text) + total


def _header(handle, path) -> dict[str, TensorInfo]:
    """The header of an open file, checked against the file's size."""
    size = os.fstat(handle.fileno()).st_size
    count = bytearray(8)
    _fill(handle, memoryview(count), path)
    length = struct.unpack("<Q", count)[0]
    if length > min(_HEADER_LIMIT, size - 8):
        raise ValueError(f"{path} is not a safetensors file (header of {length} bytes in a file of {size})")
    text = bytearray(length)
    _fill(handle, memoryview(text), path)
    try:
        entries = json.loads(text)
    except ValueError as error:
        raise ValueError(f"{path} is not a safetensors file (its header is not JSON: {error})") from None
    if not isinstance(entries, dict):
        raise ValueError(f"{path} is not a safetensors file (its header is not a JSON object)")
    base = 8 + length
    found: list[tuple[str, TensorInfo]] = []
    for name, entry in entries.items():
        if name == "__metadata__":
            continue
        try:
            code, shape = entry["dtype"], tuple(int(side) for side in entry["shape"])
            start, end = (int(offset) for offset in entry["data_offsets"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: the header's entry of {name!r} cannot be read") from None
        if code not in DTYPES:
            raise ValueError(f"{path}: {name} has dtype {code}, which is not supported ({sorted(DTYPES)})")
        elements = 1
        for side in shape:
            elements *= side
        if min(shape, default=0) < 0 or start < 0 or end - start != elements * _ITEM[code] or base + end > size:
            raise ValueError(f"{path}: {name} ({code} {list(shape)}, bytes {start}-{end}) does not fit its own "
                             f"shape or the file ({size} bytes, data from {base})")
        found.append((name, TensorInfo(code, shape, base + start, base + end)))
    found.sort(key=lambda item: (item[1].start, item[1].end, item[0]))
    return dict(found)


def _select(header: dict[str, TensorInfo], names: Iterable[str] | None, path) -> list[tuple[str, TensorInfo]]:
    """The header's entries for `names` (all when None), in file order."""
    if names is None:
        return list(header.items())
    wanted = set(names)
    unknown = sorted(wanted - header.keys())
    if unknown:
        raise KeyError(f"{path} has no tensor {unknown[0]!r}" + (f" (and {len(unknown) - 1} more)"
                                                                 if len(unknown) > 1 else ""))
    return [(name, info) for name, info in header.items() if name in wanted]


def _bytes_of(buffer: torch.Tensor) -> memoryview:
    """The buffer's memory for the file calls (the tensors handed out are views of the tensor itself)."""
    if (not isinstance(buffer, torch.Tensor) or buffer.dtype != torch.uint8 or buffer.dim() != 1
            or buffer.device.type != "cpu" or not buffer.is_contiguous()):
        raise ValueError("the buffer must be a contiguous 1-D uint8 CPU tensor")
    return memoryview(buffer.numpy())


def _pieces(handle, info: TensorInfo, buffer: torch.Tensor, raw: memoryview, path) -> Iterator[torch.Tensor]:
    """A tensor larger than the buffer, read buffer-full by buffer-full."""
    size = len(raw) - len(raw) % info.itemsize
    if not size:
        raise ValueError(f"a buffer of {len(raw)} bytes cannot hold one {info.dtype} element")
    left = info.nbytes
    while left:
        now = min(size, left)
        _fill(handle, raw[:now], path)
        left -= now
        yield buffer[:now]


def _fill(handle, view: memoryview, path) -> None:
    """Read exactly len(view) bytes (one call may return fewer)."""
    got = 0
    while got < len(view):
        count = handle.readinto(view[got:])
        if not count:
            raise EOFError(f"{path} ended {len(view) - got} bytes early")
        got += count


def _drain(handle, view: memoryview) -> None:
    """Write all of `view` (one call may take fewer bytes)."""
    sent = 0
    while sent < len(view):
        sent += handle.write(view[sent:])
