"""The flows' token-sized linear layers as int8 tensor-core GEMMs (W8A8), behind the switch PIXAL3D_INT8.

The four flow transformers (ss, lr, hr, tex: 30 blocks of 1536 channels) spend half of a detail pass at 1536, and
three quarters of an ss or lr pass, in cuBLAS bf16 GEMMs that already run at the card's bf16 tensor peak (26
TFLOPS on an RTX 3060, fp32 accumulation). The same card multiplies int8 by int8 into int32 at 84-88 TOPS
(modules/int8mm.py: Triton GEMMs with the scaling, bias and GELU fused into the epilogue, and the quantisers; the
round I speed study measured the flows' linears 2.5-3.2x faster, a whole block 2.7x, an hr pass at 21.5k tokens
4.06 -> ~2.7 s). The cost is the quantisation of the activations: a symmetric int8 code per value with one scale
per row (token), or one per row and 128 input channels ("group" scales, closer to the exact value), and the
weights quantised once at load with one scale per output channel. The round I quality study measured, with the
samplers' amplification, how far the final sample moves against the accepted yardsticks (a 1-ulp nudge of the
noise, the fused kernels, SageAttention): per-row scales on every layer move it far outside that class; group
scales on the layers that carry the error (to_qkv and the MLP) with per-row scales on the 1536 -> 1536 layers
stay nearest to it; `proj_linear`'s input has two channels ~25x the others and needs a per-channel smoothing
(SmoothQuant) or bf16; the ss and lr flows (whose samples a small change flips the most: the sparse structure's
occupancy) are kept in bf16. Which flows and layers run int8, and how, is a POLICY (below); the whole-batch
quality run decided what ships: "detail" (the detail flows hr and tex: group scales on to_qkv and the MLP, per-row
on the three 1536 -> 1536 layers, proj_linear and the ss and lr flows in bf16), on by default.

The switch is the environment variable PIXAL3D_INT8, read when `convert` runs (set_mode() changes it in a running
process):

    "on", "1", "true", "yes"     the policy DEFAULT_POLICY, after the kernels' self-check
    "off", "0", "false", "no"    the bf16 GEMMs, as before: the same numbers bit for bit
    a policy's name              that policy (POLICIES)
    unset, ""                    DEFAULT (below)
    anything else                off, with a warning

Converted layers: `Int8Linear` replaces an nn.Linear / SparseLinear (int8 weight [N, K], fp32 channel scales,
fp32 bias) and `Int8MLP` the feed-forward's (fc1, GELU, fc2) (fc1's GELU in its epilogue, fc2's input quantised
from it without a bf16 round trip of the 8192-wide hidden layer). Both take a tensor [..., K] or a VarLenTensor /
SparseTensor and answer in the same form and dtype; off the card, under autograd or autocast they fall back to
F.linear on the dequantised weight (not the bit-exact bf16 layer: the weight has been quantised). The weights take
half the memory they did (a flow's linears: 2.5 GB -> 1.3 GB), which the loader's `size_mib` reports and the
engine's room tests use.

Before the first int8 GEMM of a process the kernels run once on small tables against an fp64 reference of the same
int8 operands (int8mm.self_check); a card below sm_80 (Triton's int8 mma.sync path; the hub supports none older),
a missing Triton or a wrong answer turns the path off with an `[INT8]` warning, and the flows keep their bf16
layers. The kernels compile on first use (~6 s for the set with a cold Triton cache, 0.4 s warm; one compilation
serves every token count).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import int8mm

DEFAULT = True              # what an unset PIXAL3D_INT8 means: the policy DEFAULT_POLICY (round I: the five test props
#                             at 1536 made with it against the bf16 run: Chamfer 0.03-0.08 % of the bounding-box diagonal,
#                             at the sampling floor; the base colour at matched surface points 3-11 levels apart against
#                             13-31 for the accepted SageAttention change; renders alike; the batch 954 -> 760 s)
DEFAULT_POLICY = "detail"   # what "on" means
MIN_CAPABILITY = (8, 0)
GROUP = int8mm.GROUP        # 128 input channels per group scale

# A policy: flow kind -> {layer type: how}. Layer types: qkv (self to_qkv), out (self to_out), xq (cross to_q),
# xout (cross to_out), fc1 / fc2 (the MLP, converted together), proj (proj_linear). How: "rows" (one scale per
# token), "group" (one per token and GROUP input channels), absent = bf16. Flow kinds absent = bf16.
POLICIES: dict[str, dict[str, dict[str, str]]] = {
    # the detail flows only; group scales where the quality study saw the error, per-row on the 1536 -> 1536 layers
    "detail": {kind: {"qkv": "group", "fc1": "group", "fc2": "group", "out": "rows", "xq": "rows", "xout": "rows"}
               for kind in ("hr", "tex")},
    # the detail flows' three 1536 -> 1536 layers only: the error of the fused kernels' class, a quarter of the gain
    "safe": {kind: {"out": "rows", "xq": "rows", "xout": "rows"} for kind in ("hr", "tex")},
    # every flow, every layer, per-row scales: the fastest and the worst (experiments)
    "all": {kind: {"qkv": "rows", "fc1": "rows", "fc2": "rows", "out": "rows", "xq": "rows", "xout": "rows",
                   "proj": "rows"} for kind in ("ss", "lr", "hr", "tex")},
    # every flow with the detail policy's layers (experiments)
    "every": {kind: {"qkv": "group", "fc1": "group", "fc2": "group", "out": "rows", "xq": "rows", "xout": "rows"}
              for kind in ("ss", "lr", "hr", "tex")},
}

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}
_warned: set = set()
_mode: Optional[str] = None         # set_mode's answer (a policy name or "off"), when it was called
_state: Optional[bool] = None       # None: not tried yet; True: the kernels passed on this card; False: off
KERNEL = "int8mm: Triton int8 GEMMs, the epilogue fused"

_ATTENTION = ("SparseMultiHeadAttention", "MultiHeadAttention")
_FFN = ("SparseFeedForwardNet", "FeedForwardNet")
_PROJECT = ("SparseProjectAttention", "ProjectAttention", "SparseGatedProjectAttention", "GatedProjectAttention")


def from_env() -> Optional[str]:
    """PIXAL3D_INT8 as a policy name, or None for off."""
    text = os.environ.get("PIXAL3D_INT8", "").strip().lower()
    if text == "":
        return DEFAULT_POLICY if DEFAULT else None
    if text in _ON:
        return DEFAULT_POLICY
    if text in _OFF:
        return None
    if text in POLICIES:
        return text
    if text not in _warned:
        _warned.add(text)
        print(f"[INT8] PIXAL3D_INT8={text!r} is neither on, off nor a policy ({', '.join(POLICIES)}): the int8 "
              f"GEMMs stay off", flush=True)
    return None


def policy() -> Optional[str]:
    """The policy `convert` applies (set_mode's, else the environment's), or None for off."""
    if _mode is not None:
        return None if _mode == "off" else _mode
    return from_env()


def set_mode(mode: Optional[str | bool]) -> None:
    """A policy name, True (DEFAULT_POLICY), False / "off", or None (back to the environment's answer), for the
    networks converted from now on; networks already converted stay as they are."""
    global _mode
    if mode is None:
        _mode = None
    elif mode is True:
        _mode = DEFAULT_POLICY
    elif mode is False:
        _mode = "off"
    else:
        if mode != "off" and mode not in POLICIES:
            raise ValueError(f"unknown int8 policy {mode!r}: {', '.join(POLICIES)} or off")
        _mode = mode


# =====================================================================================================================
# The modules
# =====================================================================================================================

def _feats_in(x: Any) -> torch.Tensor:
    return x.feats if hasattr(x, "feats") else x


def _usable(x: torch.Tensor, K: int) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == K and x.numel() > 0
            and not torch.is_autocast_enabled() and not torch.is_grad_enabled() and bool(_state))


class Int8Linear(nn.Module):
    """An nn.Linear's work as an int8 GEMM (int8mm.int8_linear): `how` "rows" or "group"."""

    def __init__(self, source: nn.Linear, how: str = "rows"):
        super().__init__()
        w = source.weight.detach()
        N, K = w.shape
        if K % 16 or N % 16:
            raise ValueError(f"int8 GEMMs need K and N in multiples of 16, not {K} x {N}")
        if how not in ("rows", "group"):
            raise ValueError(f"how is rows or group, not {how!r}")
        wq, ws = int8mm.quantize_weight(w)
        self.register_buffer("weight_q", wq)                        # int8 [N, K]
        self.register_buffer("scale", ws)                           # fp32 [N]
        self.register_buffer("bias", source.bias.detach().float().contiguous() if source.bias is not None else None)
        self.in_features, self.out_features, self.how, self.dtype = K, N, how, w.dtype

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"int8 {self.how}")

    def dequantized(self) -> torch.Tensor:
        return (self.weight_q.float() * self.scale[:, None]).to(self.dtype)

    def forward(self, x: Any) -> Any:
        if hasattr(x, "feats"):
            return x.replace(self._forward(x.feats))
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _usable(x, self.in_features):
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, self.dequantized(), bias)
        return int8mm.int8_linear(x, self.weight_q, self.scale, self.bias,
                                  path="group" if self.how == "group" else "triton")


class Int8MLP(nn.Module):
    """A feed-forward's (fc1, GELU tanh, fc2) as int8mm.int8_mlp: fc1 with its GELU in the epilogue, fc2's input
    quantised from it. `how` "rows": per-row scales on both inputs; "group": group scales on both (fc1's input
    quantised in groups, fc1's epilogue writes fc2's input in groups, the hidden layer never in bf16)."""

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, how: str = "rows"):
        super().__init__()
        for name, layer in (("fc1", fc1), ("fc2", fc2)):
            N, K = layer.weight.shape
            if K % 16 or N % 16 or (how == "group" and N % GROUP):
                raise ValueError(f"{name}: int8 GEMMs need K and N in multiples of 16 (and of {GROUP} in groups), "
                                 f"not {K} x {N}")
        if how not in ("rows", "group"):
            raise ValueError(f"how is rows or group, not {how!r}")
        w1q, w1s = int8mm.quantize_weight(fc1.weight.detach())
        w2q, w2s = int8mm.quantize_weight(fc2.weight.detach())
        self.register_buffer("w1_q", w1q)
        self.register_buffer("w1_s", w1s)
        self.register_buffer("b1", fc1.bias.detach().float().contiguous() if fc1.bias is not None else None)
        self.register_buffer("w2_q", w2q)
        self.register_buffer("w2_s", w2s)
        self.register_buffer("b2", fc2.bias.detach().float().contiguous() if fc2.bias is not None else None)
        self.in_features, self.hidden, self.out_features = fc1.weight.shape[1], fc1.weight.shape[0], fc2.weight.shape[0]
        self.how, self.dtype = how, fc1.weight.dtype

    def extra_repr(self) -> str:
        return f"{self.in_features} -> {self.hidden} -> {self.out_features}, GELU(tanh), int8 {self.how}"

    def forward(self, x: Any) -> Any:
        if hasattr(x, "feats"):
            return x.replace(self._forward(x.feats))
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.in_features
        if not _usable(x, K):
            w1 = (self.w1_q.float() * self.w1_s[:, None]).to(x.dtype)
            w2 = (self.w2_q.float() * self.w2_s[:, None]).to(x.dtype)
            h = F.gelu(F.linear(x, w1, self.b1.to(x.dtype) if self.b1 is not None else None), approximate="tanh")
            return F.linear(h, w2, self.b2.to(x.dtype) if self.b2 is not None else None)
        flat = x.reshape(-1, K)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        w1, w2 = (self.w1_q, self.w1_s, self.b1), (self.w2_q, self.w2_s, self.b2)
        if self.how == "group":
            y = int8mm.int8_mlp(int8mm.quantize_rows(flat, GROUP), w1, w2, path="group")
        else:
            y = int8mm.int8_mlp(flat, w1, w2, path="rows")
        return y.view(*x.shape[:-1], self.out_features)


# =====================================================================================================================
# The conversion
# =====================================================================================================================

def _targets(module: nn.Module, rules: dict[str, str]) -> list[tuple[str, Any, Any, str]]:
    """(layer type, container, key, how) of the token-sized linears directly inside `module` that `rules` convert."""
    name = type(module).__name__
    found: list[tuple[str, Any, Any, str]] = []
    if name in _ATTENTION:
        cross = getattr(module, "_type", "self") == "cross"
        for attr, kind in (("to_qkv", "qkv"), ("to_q", "xq"), ("to_out", "xout" if cross else "out")):
            if kind in rules and isinstance(getattr(module, attr, None), nn.Linear):
                found.append((kind, module, attr, rules[kind]))
    elif name in _PROJECT:
        if "proj" in rules and isinstance(getattr(module, "proj_linear", None), nn.Linear):
            found.append(("proj", module, "proj_linear", rules["proj"]))
    return found


def convert(model: nn.Module, kind: str = "") -> nn.Module:
    """The model (a flow of `kind`: ss, lr, hr or tex) with its token-sized linears as int8 layers where the policy
    says so, when the switch is on and the card and the self-check allow it; the model unchanged otherwise. Called
    by the staged loader after a flow network is loaded (its weights on the card in bf16; each bf16 weight is freed
    as it is replaced)."""
    name = policy()
    if name is None:
        return model
    rules = POLICIES[name].get(kind)
    if not rules:
        return model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return model
    if device.type != "cuda" or not available(device):
        return model
    before = _bytes(model)
    counts: dict[str, int] = {}
    with torch.no_grad():
        for module in list(model.modules()):
            mlp = getattr(module, "mlp", None)
            if (type(module).__name__ in _FFN and "fc1" in rules and "fc2" in rules and isinstance(mlp, nn.Sequential)
                    and len(mlp) == 3 and isinstance(mlp[0], nn.Linear) and isinstance(mlp[2], nn.Linear)
                    and type(mlp[1]).__name__ in ("SparseGELU", "GELU")):
                how = rules["fc1"]
                module.mlp = Int8MLP(mlp[0], mlp[2], how)
                counts["mlp"] = counts.get("mlp", 0) + 1
                del mlp
                continue
            for layer, container, key, how in _targets(module, rules):
                old = getattr(container, key)
                setattr(container, key, Int8Linear(old, how))
                counts[layer] = counts.get(layer, 0) + 1
                del old
    if counts:
        print(f"[INT8] {type(model).__name__} ({kind}, policy {name}): "
              + ", ".join(f"{k} x{v}" for k, v in counts.items())
              + f" as int8 GEMMs ({KERNEL}; PIXAL3D_INT8), {before / 2**20:,.0f} -> {_bytes(model) / 2**20:,.0f} MiB "
                f"of weights", flush=True)
    return model


def _bytes(model: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


# =====================================================================================================================
# The self-check
# =====================================================================================================================

def available(device: torch.device) -> bool:
    """Whether the int8 kernels run on this card (tried once: int8mm.ready)."""
    global _state
    if _state is None:
        try:
            major, minor = torch.cuda.get_device_capability(device)
            if (major, minor) < MIN_CAPABILITY:
                raise RuntimeError(f"the card is sm_{major}{minor}; the int8 GEMMs are used from "
                                   f"sm_{MIN_CAPABILITY[0]}{MIN_CAPABILITY[1]} on")
            ok, why = int8mm.ready(device)
            if not ok:
                raise RuntimeError(why)
            torch.cuda.synchronize(device)
        except Exception as exc:        # noqa: BLE001 - whatever the kernels, Triton or the card raise: bf16 serves
            _state = False
            print(f"[INT8] the int8 GEMMs do not run on this card ({type(exc).__name__}: {str(exc)[:300]}); the "
                  f"flows keep their bf16 layers", flush=True)
            return False
        _state = True
        print(f"[INT8] the flows' token-sized linears can run as int8 GEMMs: {KERNEL} (PIXAL3D_INT8)", flush=True)
    return bool(_state)
