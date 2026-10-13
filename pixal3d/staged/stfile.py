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
    write_streamed(path, layout, pieces, metadata=...)  # tensor by tensor, piece by piece (weights.py's copies)
    metadata = read_metadata(path)                      # the header's "__metadata__"
    digest = header_sha256(path)                        # a fingerprint of the header (one small read)

The format: 8 bytes (little-endian count N), N bytes of JSON {name: {"dtype", "shape", "data_offsets": [a, b]}},
then the tensors' bytes, each at a + 8 + N. Complex64 is stored as pairs of float32 (real, imaginary). The JSON
may also hold "__metadata__": {text: text}, which is no tensor.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
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


def read_metadata(path: str | os.PathLike) -> dict[str, str]:
    """The header's "__metadata__" ({} when it has none). Reads the header only and checks it as read_header
    does, so a file whose tensors do not fit it raises ValueError here too."""
    with open(path, "rb", buffering=0) as handle:
        entries, base = _entries(handle, path)
        _infos(entries, base, os.fstat(handle.fileno()).st_size, path)
    metadata = entries.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(key, str) and isinstance(value, str)
                                                 for key, value in metadata.items()):
        raise ValueError(f"{path}: the header's __metadata__ is not a map of texts")
    return dict(metadata)


def header_sha256(path: str | os.PathLike) -> str:
    """SHA-256 of the file's first 8 + N bytes, its header's length and the header: a fingerprint of the names,
    dtypes, shapes and places of its tensors that costs one small read."""
    with open(path, "rb", buffering=0) as handle:
        size = os.fstat(handle.fileno()).st_size
        count = bytearray(8)
        _fill(handle, memoryview(count), path)
        length = struct.unpack("<Q", count)[0]
        if length > min(_HEADER_LIMIT, size - 8):
            raise ValueError(f"{path} is not a safetensors file (header of {length} bytes in a file of {size})")
        text = bytearray(length)
        _fill(handle, memoryview(text), path)
    return hashlib.sha256(count + text).hexdigest()


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
    text = _header_text(header)
    with _replacing(path, fsync) as handle:
        _drain(handle, memoryview(text))
        for name, code, tensor, size in entries:
            _write_memory(handle, tensor, size)
    return len(text) + total


def write_streamed(path: str | os.PathLike, layout: Iterable[tuple[str, torch.dtype, tuple[int, ...]]],
                   pieces: Iterable[tuple[str, torch.Tensor]], *, metadata: dict[str, str] | None = None,
                   fsync: bool = False) -> int:
    """Write a safetensors file whose tensors arrive piece by piece, so that only one piece need be in memory at a
    time; return its size in bytes.

    `layout` gives every tensor (name, dtype, shape) in the order they take in the file, one after the other
    without gaps: a tensor then starts on a multiple of its element size only when the ones before it leave it
    there (none of the readers here needs that). `pieces` yields (name, CPU tensor) in the layout's order, each
    tensor being the next elements of the tensor of that name, in its dtype, as many as the piece holds (a tensor
    of no elements needs no piece). `metadata` becomes the header's "__metadata__". Written through <path>.tmp as
    `write` is; ValueError when the pieces do not fill the layout exactly, and then nothing is left behind."""
    order: list[tuple[str, torch.dtype, int]] = []
    header: dict[str, dict] = {}
    if metadata:
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
            raise ValueError("metadata is a map of texts")
        header["__metadata__"] = dict(metadata)
    total = 0
    for name, dtype, shape in layout:
        code = _CODE.get(dtype)
        if not isinstance(name, str) or name == "__metadata__" or name in header:
            raise ValueError(f"{name!r} cannot be a tensor's name here")
        if code is None:
            raise ValueError(f"{name}: {dtype} cannot be written (supported: {sorted(DTYPES)})")
        elements = 1
        for side in shape:
            elements *= int(side)
        size = elements * _ITEM[code]
        header[name] = {"dtype": code, "shape": [int(side) for side in shape], "data_offsets": [total, total + size]}
        order.append((name, dtype, size))
        total += size
    text = _header_text(header)

    with _replacing(path, fsync) as handle:
        _drain(handle, memoryview(text))
        index, left = -1, 0
        for name, piece in pieces:
            while index < 0 or name != order[index][0]:
                if left:
                    raise ValueError(f"{order[index][0]}: {left} bytes were still to come when {name} began")
                index += 1
                if index == len(order):
                    raise ValueError(f"{name} is not in the layout, or comes out of its order")
                left = order[index][2]
            if (not isinstance(piece, torch.Tensor) or piece.dtype != order[index][1]
                    or piece.device.type != "cpu" or piece.layout != torch.strided):
                raise ValueError(f"{name}: a piece must be a dense CPU tensor of {order[index][1]}")
            if piece.is_conj() or piece.is_neg():
                piece = piece.resolve_conj().resolve_neg()
            piece = piece.contiguous()
            size = piece.numel() * piece.element_size()
            if size > left:
                raise ValueError(f"{name}: more elements came than its shape holds")
            _write_memory(handle, piece, size)
            left -= size
        if index >= 0 and left:
            raise ValueError(f"the pieces ended {left} bytes before the end of {order[index][0]}")
        rest = [entry[0] for entry in order[index + 1:] if entry[2]]
        if rest:
            raise ValueError(f"the pieces ended before {rest[0]}")
    return len(text) + total


def _header_text(header: dict) -> bytes:
    """The 8-byte length and the JSON of a header, padded with spaces to a multiple of 8 bytes."""
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text


@contextlib.contextmanager
def _replacing(path: str | os.PathLike, fsync: bool) -> Iterator:
    """An open <path>.tmp to write; renamed over `path` when the block ends well, removed when it does not."""
    partial = f"{os.fspath(path)}.tmp"
    try:
        with open(partial, "wb", buffering=0) as handle:
            yield handle
            if fsync:
                os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        try:
            os.unlink(partial)
        except OSError:
            pass
        raise


def _write_memory(handle, tensor: torch.Tensor, size: int) -> None:
    """The `size` bytes of a contiguous CPU tensor's memory, straight out of it."""
    if size:
        _drain(handle, memoryview((ctypes.c_ubyte * size).from_address(tensor.data_ptr())))


def _header(handle, path) -> dict[str, TensorInfo]:
    """The header of an open file, checked against the file's size."""
    entries, base = _entries(handle, path)
    return _infos(entries, base, os.fstat(handle.fileno()).st_size, path)


def _entries(handle, path) -> tuple[dict, int]:
    """The header's JSON object of an open file, and where the tensors' bytes begin."""
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
    return entries, 8 + length


def _infos(entries: dict, base: int, size: int, path) -> dict[str, TensorInfo]:
    """name -> TensorInfo of a header's entries, sorted by `start`, each checked against the file's size."""
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
