"""The flows' token-sized linear layers as int8 tensor-core GEMMs (W8A8), behind the switch PIXAL3D_INT8.

The four flow transformers (ss, lr, hr, tex: 30 blocks of 1536 channels) spend about half of a detail pass at 1536,
and ~85 % of an ss or lr pass, in cuBLAS bf16 GEMMs that already run at the card's bf16 tensor peak (fp32
accumulation). GeForce cards from Turing on multiply int8 by int8 into int32 at about four times that rate. With
the switch on, every linear layer that runs over the prop's tokens - per block: the self-attention's `to_qkv` and
`to_out`, the cross-attention's `to_q` and `to_out`, the MLP's two layers; `proj_linear` only when PROJ says so -
is replaced after the network is loaded (`convert`, called by the staged loader) by an Int8Linear: the weight
quantised once, symmetrically, with one scale per output channel; the activations quantised at every call with one
scale per row (token); the int32 result scaled back, the bias added and the row rounded once to bf16. The layers
that run over a handful of tokens (the cross-attention's `to_kv`, the adaLN, the input and output layers) stay in
bf16. The weights take half the memory they did (a flow's linears: 2.4 GB -> 1.2 GB), which the loader's
`size_mib` reports and the engine's room tests use.

The switch is the environment variable PIXAL3D_INT8, read when `convert` runs (set_mode() changes it in a running
process):

    "on", "1", "true", "yes"     int8 GEMMs where they apply, after the self-check
    "off", "0", "false", "no"    the bf16 GEMMs, as before: the same numbers bit for bit
    unset, ""                    DEFAULT (below)
    anything else                off, with a warning

The kernels: `torch._int_mm` (cuBLASLt's int8 IMMA GEMM, int32 out) with the quantisation and the epilogue as
plain tensor ops over row chunks of ROWS tokens (the fused Triton kernels of the speed study replace these when
they are in). The quantisation is round-to-nearest, clamped to [-127, 127]; the scales fp32; a zero row or weight
channel gets the scale 1 (its values are zero either way).

Before the first int8 GEMM of a process the path runs once on a small random table against an fp32 reference of
the same quantised operands (`_check`): the integer part must match exactly and the epilogue within bf16
rounding. A card older than sm_80 (the hub supports none), a missing `_int_mm` or a wrong answer turns the path
off with an `[INT8]` warning, and the flows keep their bf16 layers.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT = False     # what an unset PIXAL3D_INT8 means (off until the whole-batch quality run accepts the path)
PROJ = "bf16"       # proj_linear (2048 -> 1536, 7 % of a block's GEMM work; its input has two channels ~25x the
#                     others, which per-row scales cannot follow: 3-9 % layer error, against ~1 % elsewhere):
#                     "bf16" keeps it as it is, "int8" converts it like the rest
ROWS = 4096         # rows per chunk of the tensor-op path (bounds the int32 and fp32 temporaries: 4096 x 8192 x 4 B)
MIN_ROWS = 32       # torch._int_mm wants more than 16 rows: fewer are padded up to this
MIN_CAPABILITY = (8, 0)

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}
_warned: set = set()
_mode: Optional[bool] = None        # set_mode's answer, when it was called; else the environment decides
_state: Optional[bool] = None       # None: not tried yet; True: passed its self-check on this card; False: off
KERNEL = ""                         # the GEMM path in use, for reports

# Which attributes of which module classes are the token-sized linears (by class name: this module must not import
# the sparse and dense attention modules, which import it)
_ATTENTION = ("SparseMultiHeadAttention", "MultiHeadAttention")
_FFN = ("SparseFeedForwardNet", "FeedForwardNet")
_PROJECT = ("SparseProjectAttention", "ProjectAttention", "SparseGatedProjectAttention", "GatedProjectAttention")


def from_env() -> bool:
    """PIXAL3D_INT8 as a bool."""
    text = os.environ.get("PIXAL3D_INT8", "").strip().lower()
    if text == "":
        return DEFAULT
    if text in _ON:
        return True
    if text not in _OFF and text not in _warned:
        _warned.add(text)
        print(f"[INT8] PIXAL3D_INT8={text!r} is neither on nor off: the int8 GEMMs stay off", flush=True)
    return False


def enabled() -> bool:
    """Whether `convert` converts: set_mode's answer, else the environment's."""
    return from_env() if _mode is None else _mode


def set_mode(on: Optional[bool]) -> None:
    """On or off for the networks converted from now on (None: back to the environment's answer). Networks already
    converted stay as they are."""
    global _mode
    _mode = None if on is None else bool(on)


# =====================================================================================================================
# The quantisation and the GEMM (tensor ops; the fused kernels replace the insides of these three functions)
# =====================================================================================================================

def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [N, K] (bf16 or fp32) -> (int8 [N, K], fp32 scales [N]): one symmetric scale per output channel."""
    wf = w.detach().float()
    amax = wf.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [M, K] (bf16) -> (int8 [M, K], fp32 scales [M]): one symmetric scale per row."""
    amax = x.abs().amax(dim=1).float()
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(x.float() / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def gemm(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, bias: Optional[torch.Tensor],
         out_dtype: torch.dtype) -> torch.Tensor:
    """(xq [M, K] int8, row scales xs [M]) times (wq [N, K] int8, channel scales ws [N])^T, plus bias [N] (fp32),
    rounded once to out_dtype: [M, N]."""
    acc = torch._int_mm(xq, wq.t())                 # int32 [M, N], exact
    y = acc.float().mul_(xs[:, None]).mul_(ws[None, :])
    if bias is not None:
        y.add_(bias[None, :])
    return y.to(out_dtype)


def linear(x: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """x [M, K] (bf16, contiguous) through the int8 path in chunks of ROWS rows: [M, N] in x's dtype."""
    M, K = x.shape
    if M < MIN_ROWS:
        padded = torch.zeros((MIN_ROWS, K), dtype=x.dtype, device=x.device)
        padded[:M] = x
        return linear(padded, wq, ws, bias)[:M]
    if M <= ROWS:
        xq, xs = quantize_rows(x)
        return gemm(xq, xs, wq, ws, bias, x.dtype)
    out = torch.empty((M, wq.shape[0]), dtype=x.dtype, device=x.device)
    for start in range(0, M, ROWS):
        piece = x[start:start + ROWS]
        if piece.shape[0] < MIN_ROWS:               # a short last chunk: pad it rather than call _int_mm short
            out[start:] = linear(piece, wq, ws, bias)
            break
        xq, xs = quantize_rows(piece)
        out[start:start + ROWS] = gemm(xq, xs, wq, ws, bias, x.dtype)
    return out


# =====================================================================================================================
# The module and the conversion
# =====================================================================================================================

class Int8Linear(nn.Module):
    """An nn.Linear's work as an int8 GEMM. Takes a tensor [..., K] or a VarLenTensor / SparseTensor (whose feats
    are [M, K]), as the layers it replaces do, and answers in the same form and dtype."""

    def __init__(self, source: nn.Linear):
        super().__init__()
        w = source.weight.detach()
        N, K = w.shape
        if K % 8 or N % 8:
            raise ValueError(f"int8 GEMMs need K and N in multiples of 8, not {K} x {N}")
        wq, ws = quantize_weight(w)
        self.register_buffer("weight_q", wq)                        # int8 [N, K]
        self.register_buffer("scale", ws)                           # fp32 [N]
        self.register_buffer("bias", source.bias.detach().float().contiguous() if source.bias is not None else None)
        self.in_features, self.out_features = K, N
        self.dtype = w.dtype

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, int8"

    def dequantized(self) -> torch.Tensor:
        """The weight as the layer runs it, in the source's dtype [N, K] (the fallback path's weight)."""
        return (self.weight_q.float() * self.scale[:, None]).to(self.dtype)

    def forward(self, x: Any) -> Any:
        if hasattr(x, "feats"):                                     # VarLenTensor / SparseTensor
            return x.replace(self._forward(x.feats))
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        K, N = self.in_features, self.out_features
        if (not x.is_cuda or x.dtype != self.dtype or x.shape[-1] != K or torch.is_autocast_enabled()
                or torch.is_grad_enabled() or x.numel() == 0):
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, self.dequantized(), bias)
        flat = x.reshape(-1, K)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        return linear(flat, self.weight_q, self.scale, self.bias).view(*x.shape[:-1], N)


def _targets(module: nn.Module) -> list[tuple[Any, Any]]:
    """(container, key) of the token-sized linears directly inside `module`, by its class name."""
    name = type(module).__name__
    found: list[tuple[Any, Any]] = []
    if name in _ATTENTION:
        for attr in ("to_qkv", "to_q", "to_out"):
            if isinstance(getattr(module, attr, None), nn.Linear):
                found.append((module, attr))
    elif name in _FFN:
        mlp = getattr(module, "mlp", None)
        if isinstance(mlp, nn.Sequential):
            for index, layer in enumerate(mlp):
                if isinstance(layer, nn.Linear):
                    found.append((mlp, index))
    elif name in _PROJECT and PROJ == "int8":
        if isinstance(getattr(module, "proj_linear", None), nn.Linear):
            found.append((module, "proj_linear"))
    return found


def convert(model: nn.Module) -> nn.Module:
    """The model with its token-sized linears as Int8Linear layers, when the switch is on and the card and the
    self-check allow it; the model unchanged otherwise. Called by the staged loader after a flow network is loaded
    (the weights on the card in their bf16 form; each layer's bf16 weight is freed as it is replaced)."""
    if not enabled():
        return model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return model
    if device.type != "cuda" or not available(device):
        return model
    before = _bytes(model)
    count = 0
    with torch.no_grad():
        for module in list(model.modules()):
            for container, key in _targets(module):
                old = container[key] if isinstance(key, int) else getattr(container, key)
                new = Int8Linear(old)
                if isinstance(key, int):
                    container[key] = new
                else:
                    setattr(container, key, new)
                count += 1
                del old
    if count:
        print(f"[INT8] {type(model).__name__}: {count} linear layers as int8 GEMMs ({KERNEL}; PIXAL3D_INT8), "
              f"{before / 2**20:,.0f} -> {_bytes(model) / 2**20:,.0f} MiB of weights", flush=True)
    return model


def _bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


# =====================================================================================================================
# The self-check
# =====================================================================================================================

def available(device: torch.device) -> bool:
    """Whether the int8 path runs on this card (tried once, `_check`)."""
    if _state is None:
        _check(device)
    return bool(_state)


def _check(device: torch.device) -> bool:
    """The path once on a random table against an fp32 reference of the same quantised operands: the integer GEMM
    exactly, the whole layer within bf16 rounding. Anything else turns the path off."""
    global _state, KERNEL
    try:
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < MIN_CAPABILITY:
            raise RuntimeError(f"the card is sm_{major}{minor}; int8 tensor-core GEMMs are used from "
                               f"sm_{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]} on")
        g = torch.Generator(device=device).manual_seed(0)
        M, K, N = 1000, 256, 384
        x = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g) * 3
        x[:, 7] *= 20                                               # one outlying channel, as the flows have
        w = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g) * 0.05
        b = torch.randn((N,), device=device, dtype=torch.float32, generator=g)
        wq, ws = quantize_weight(w)
        xq, xs = quantize_rows(x)
        acc = torch._int_mm(xq, wq.t())
        want_int = (xq.float() @ wq.float().t())                   # exact in fp32 (|sum| < 2^24)
        if not torch.equal(acc.float(), want_int):
            raise RuntimeError("torch._int_mm's integer answer is wrong on this card")
        got = linear(x, wq, ws, b).float()
        want = (want_int * xs[:, None] * ws[None, :] + b[None, :])
        rel = float((got - want).norm() / want.norm())
        torch.cuda.synchronize(device)
        if not bool(torch.isfinite(got).all()) or not rel <= 1e-2:
            raise RuntimeError(f"the int8 layer's answer differs from the reference (relative L2 {rel:.3g})")
        KERNEL = "torch._int_mm + tensor-op epilogue"
    except Exception as exc:        # noqa: BLE001 - whatever torch or the card raises: the bf16 layers serve
        _state = False
        print(f"[INT8] the int8 GEMMs do not run on this card ({type(exc).__name__}: {str(exc)[:300]}); the flows "
              f"keep their bf16 layers", flush=True)
        return False
    _state = True
    print(f"[INT8] the flows' token-sized linears run as int8 GEMMs: {KERNEL} (PIXAL3D_INT8)", flush=True)
    return True
