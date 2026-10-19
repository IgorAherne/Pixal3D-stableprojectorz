"""The masked convolutions' row codes in one pass: a Triton kernel for what lean_cache.build and chunked.subset
make with 27 + 26 eager integer passes over the neighbour map.

    gray, binary = codes(neighbor_map, volume)      # the Gray code (bit v: a neighbour at offset v) and the binary
                                                    # number whose Gray code it is
    binary = binary_of(gray, volume)                # the second alone, from a Gray code

The values are the eager ops' exactly (integers: the bits are disjoint, so the kernel's sum is their OR; the binary
code is the code xor-ed with itself shifted right by 1, 2, ..., volume - 1, as the eager loop does). Measured on the
1536 golden prop's levels (round J, Spellcast3D code/devtools/pixal3d-lowmem/round-j/parts/lean_codes_check.py).

Before the first fused call of a process the kernels run once on a small random table against the eager ops; a
card or a Triton that cannot compile or run them, or runs them wrong, makes every later call take the eager ops
(with one warning). ENABLED = False is the way back.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

ENABLED = True
BLOCK = 256
_kernels = None          # (codes, binary) Triton kernels, built at first use
_state: Optional[bool] = None   # None: not tried yet; True: checked; False: eager from now on


def eager_codes(neighbor_map: torch.Tensor, volume: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The eager ops (lean_cache.build's loops), the reference."""
    neighbors = neighbor_map.view(torch.int32)
    gray = torch.zeros(neighbor_map.shape[0], dtype=torch.int32, device=neighbor_map.device)
    for v in range(volume):
        gray.bitwise_or_((neighbors[:, v] != -1).to(torch.int32).bitwise_left_shift_(v))
    return gray, eager_binary(gray, volume)


def eager_binary(gray: torch.Tensor, volume: int) -> torch.Tensor:
    binary = gray.clone()
    for v in range(1, volume):
        binary.bitwise_xor_(gray >> v)
    return binary


def _build():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def codes_kernel(nmap, gray_out, binary_out, n, V: tl.constexpr, VP: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rmask = rows < n
        cols = tl.arange(0, VP)
        nb = tl.load(nmap + rows[:, None].to(tl.int64) * V + cols[None, :], mask=rmask[:, None] & (cols < V)[None, :],
                     other=-1)
        gray = tl.sum(tl.where(nb != -1, (1 << cols)[None, :], 0), axis=1)
        binary = gray
        for v in tl.static_range(1, V):
            binary = binary ^ (gray >> v)
        tl.store(gray_out + rows, gray, mask=rmask)
        tl.store(binary_out + rows, binary, mask=rmask)

    @triton.jit
    def binary_kernel(gray_in, binary_out, n, V: tl.constexpr, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rmask = rows < n
        gray = tl.load(gray_in + rows, mask=rmask, other=0)
        binary = gray
        for v in tl.static_range(1, V):
            binary = binary ^ (gray >> v)
        tl.store(binary_out + rows, binary, mask=rmask)

    _kernels = (codes_kernel, binary_kernel, triton.cdiv)
    return _kernels


def _fused_codes(neighbor_map: torch.Tensor, volume: int) -> Tuple[torch.Tensor, torch.Tensor]:
    codes_kernel, _, cdiv = _build()
    nb = neighbor_map.view(torch.int32)
    n = nb.shape[0]
    gray = torch.empty(n, dtype=torch.int32, device=nb.device)
    binary = torch.empty(n, dtype=torch.int32, device=nb.device)
    if n:
        codes_kernel[(cdiv(n, BLOCK),)](nb, gray, binary, n, V=volume, VP=32, BLOCK=BLOCK)
    return gray, binary


def _fused_binary(gray: torch.Tensor, volume: int) -> torch.Tensor:
    _, binary_kernel, cdiv = _build()
    gray = gray.contiguous()
    n = gray.shape[0]
    binary = torch.empty_like(gray)
    if n:
        binary_kernel[(cdiv(n, BLOCK),)](gray, binary, n, V=volume, BLOCK=BLOCK)
    return binary


def _usable(t: torch.Tensor, volume: int) -> bool:
    global _state
    if not ENABLED or not t.is_cuda or volume > 31 or _state is False:
        return False
    if _state is None:
        try:
            g = torch.Generator(device=t.device).manual_seed(0)
            nmap = torch.randint(-1, 1 << 20, (4099, volume), device=t.device, dtype=torch.int32, generator=g)
            nmap[torch.rand(nmap.shape, device=t.device, generator=g) < 0.4] = -1
            want_g, want_b = eager_codes(nmap, volume)
            got_g, got_b = _fused_codes(nmap, volume)
            ok = (torch.equal(want_g, got_g) and torch.equal(want_b, got_b)
                  and torch.equal(eager_binary(want_g, volume), _fused_binary(want_g, volume)))
            if not ok:
                raise RuntimeError("the fused codes differ from the eager ones")
            _state = True
        except Exception as exc:        # noqa: BLE001 - the eager ops always work
            print(f"[SPARSE] the fused neighbour codes are off ({type(exc).__name__}: {str(exc)[:160]}); the eager "
                  f"ops run instead", flush=True)
            _state = False
    return _state


def codes(neighbor_map: torch.Tensor, volume: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(gray, binary) int32 [n] of a neighbour map [n, volume] (uint32 or int32; missing = 0xffffffff)."""
    if _usable(neighbor_map, volume):
        return _fused_codes(neighbor_map, volume)
    return eager_codes(neighbor_map, volume)


def binary_of(gray: torch.Tensor, volume: int) -> torch.Tensor:
    """The binary code of a Gray code [n] int32."""
    if _usable(gray, volume):
        return _fused_binary(gray, volume)
    return eager_binary(gray, volume)
