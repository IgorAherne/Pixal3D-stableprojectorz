"""The flows' token-sized linear layers as int8 tensor-core GEMMs (W8A8), behind the switch PIXAL3D_INT8.

The four flow transformers (ss, lr, hr, tex: 30 blocks of 1536 channels) spend half of a detail pass at 1536, and
three quarters of an ss or lr pass, in cuBLAS bf16 GEMMs that already run at the card's bf16 tensor peak (26
TFLOPS on an RTX 3060, fp32 accumulation). The same card multiplies int8 by int8 into int32 at 84-88 TOPS
(modules/int8mm.py: Triton GEMMs with the scaling, bias and GELU fused into the epilogue, and the quantisers; the
round I speed study measured the flows' linears 2.5-3.2x faster, a whole block 2.7x, an hr pass at 21.5k tokens
4.06 -> ~2.7 s). The cost is the quantisation of the activations: a symmetric int8 code per value with one scale
per row (token), or one per row and 128 input channels ("group" scales, closer to the exact value), and the
weights quantised once at load with one scale per output channel (or per output channel and 128 input channels).
The round I quality study measured, with the samplers' amplification, how far the final sample moves against the
accepted yardsticks (a 1-ulp nudge of the noise, the fused kernels, SageAttention): per-row scales on every layer
move it far outside that class; group scales on the layers that carry the error (to_qkv and the MLP) with per-row
scales on the 1536 -> 1536 layers stay nearest to it; `proj_linear`'s input has two channels ~25x the others and
needs a per-channel smoothing (SmoothQuant) or bf16; the ss and lr flows (whose samples a small change flips the
most: the sparse structure's occupancy) are kept in bf16. Which flows and layers run int8, and how, is a POLICY
(below); the whole-batch quality run decided what ships: "detail" (the detail flows hr and tex: group scales on
to_qkv and the MLP, per-row on the three 1536 -> 1536 layers, proj_linear and the ss and lr flows in bf16), on by
default.

Policy "cs" (round J): 128-K group scales on the activations AND the weights of every token-sized layer including
proj_linear, plus static SmoothQuant where it helps: the activation divided by a fixed per-input-channel vector s
as it is quantised and the weight multiplied by s before it is quantised (exact in real numbers; it moves the
outlier channels' range from the activations into the weights). The vectors are calibrated once per checkpoint
from golden cases (code/devtools/pixal3d-lowmem/round-j/int8cs/make_calib.py) and ship as small files in
modules/int8_calib/<checkpoint>.safetensors (fp16 vectors keyed by the layer's name, or "shared.<type>" for a
vector every block of a type uses: proj_linear's input, the conditioning, is then quantised once per pass). A
checkpoint without a file runs the policy without the smoothing (group scales only) and keeps proj_linear in bf16,
and says so in one [INT8] line. "cs-every": the same on all four flows (an experiment: ss and lr are judged end to
end).

The switch is the environment variable PIXAL3D_INT8, read when `convert` runs (set_mode() changes it in a running
process):

    "on", "1", "true", "yes"     the policy DEFAULT_POLICY, after the kernels' self-check
    "off", "0", "false", "no"    the bf16 GEMMs, as before: the same numbers bit for bit
    a policy's name              that policy (POLICIES)
    unset, ""                    DEFAULT (below)
    anything else                off, with a warning

Converted layers: `Int8Linear` replaces an nn.Linear / SparseLinear (int8 weight [N, K], fp32 scales [N] or [N,
K / 128], fp32 bias, the smoothing's 1 / s) and `Int8MLP` the feed-forward's (fc1, GELU, fc2) (fc1's GELU in its
epilogue, fc2's input quantised from it without a bf16 round trip of the 8192-wide hidden layer). Both take a
tensor [..., K] or a VarLenTensor / SparseTensor and answer in the same form and dtype; off the card, under
autograd or autocast they fall back to F.linear on the dequantised weight (not the bit-exact bf16 layer: the
weight has been quantised). The weights take half the memory they did (a flow's linears: 2.5 GB -> 1.3 GB), which
the loader's `size_mib` reports and the engine's room tests use.

Before the first int8 GEMM of a process the kernels run once on small tables against an fp64 reference of the same
int8 operands (int8mm.self_check); a card below sm_80 (Triton's int8 mma.sync path; the hub supports none older),
a missing Triton or a wrong answer turns the path off with an `[INT8]` warning, and the flows keep their bf16
layers. The kernels compile on first use (~6 s for the set with a cold Triton cache, 0.4 s warm; one compilation
serves every token count).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
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
CALIB_DIR = Path(__file__).resolve().parent / "int8_calib"
SHARE_INPUT = True          # a "shared.<type>" smoothing vector: the layer's input quantised once per pass (proj)

# The cs policies' layer rules (round J, fake-quantised whole samplings: step 1 of round-j/int8cs): the smoothing
# where it helps (to_qkv, fc1, the cross to_q, proj_linear), plain group scales on fc2 and the two to_out layers.
CS_RULES = {"qkv": "cs", "fc1": "cs", "fc2": "cg", "out": "cg", "xq": "cs", "xout": "cg", "proj": "cs"}

# A policy: flow kind -> {layer type: how}. Layer types: qkv (self to_qkv), out (self to_out), xq (cross to_q),
# xout (cross to_out), fc1 / fc2 (the MLP, converted together), proj (proj_linear). How: "rows" (one scale per
# token; weights per output channel), "group" (one per token and GROUP input channels; weights per output channel),
# "cg" (group scales on the activations and on the weights: one per output channel and GROUP input channels), "cs"
# ("cg" with the checkpoint's SmoothQuant vector; without a calibration file "cg" and proj_linear in bf16),
# absent = bf16. Flow kinds absent = bf16.
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
    # round J: group scales on both operands of every token-sized layer + the calibrated smoothing, the detail flows
    "cs": {kind: dict(CS_RULES) for kind in ("hr", "tex")},
    # the same on all four flows (an experiment: ss and lr are the structure-sensitive ones)
    "cs-every": {kind: dict(CS_RULES) for kind in ("ss", "lr", "hr", "tex")},
}
HOWS = ("rows", "group", "cg", "cs")

# layer type -> the linear's path inside a block (blocks.<i>.<path>): the calibration files' keys
PATHS = {"qkv": "self_attn.to_qkv", "out": "self_attn.to_out", "fc1": "mlp.mlp.0", "fc2": "mlp.mlp.2",
         "xq": "cross_attn.cross_attn_block.to_q", "xout": "cross_attn.cross_attn_block.to_out",
         "proj": "cross_attn.proj_linear"}

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
# The calibration (SmoothQuant vectors per checkpoint)
# =====================================================================================================================

def calib_file(checkpoint: str) -> Path:
    """Where the SmoothQuant vectors of a checkpoint (its file name without extension) lie."""
    return CALIB_DIR / f"{Path(str(checkpoint)).name}.safetensors"


def load_calib(checkpoint: str) -> Optional[dict[str, torch.Tensor]]:
    """{layer name or "shared.<type>": s (fp32 [K])} of a checkpoint, or None when it has no (readable) file. s
    multiplies the weight's input channels; the activations are divided by it."""
    if not checkpoint:
        return None
    path = calib_file(checkpoint)
    if not path.is_file():
        return None
    try:
        from safetensors.torch import load_file

        table = {name: t.float() for name, t in load_file(str(path)).items()}
    except Exception as exc:        # noqa: BLE001 - an unreadable file is a missing file
        print(f"[INT8] the calibration file {path.name} cannot be read ({type(exc).__name__}: {str(exc)[:200]})",
              flush=True)
        return None
    bad = [name for name, t in table.items() if t.dim() != 1 or not bool(torch.isfinite(t).all()) or
           not bool((t > 0).all())]
    if bad:
        print(f"[INT8] the calibration file {path.name} holds vectors that are not finite and positive "
              f"({', '.join(bad[:3])}); it is not used", flush=True)
        return None
    return table


class SharedInput:
    """One smoothing vector's 1 / s for every block of a layer type whose input is the same tensor in every block
    (proj_linear: the conditioning), and the input as quantised by the first of them in a pass: the other blocks
    take it from here. The entry holds the tensor itself (no other tensor can take its address while it is kept)
    and its version counter (an in-place change makes it stale); the flow's forward hooks drop it at the start and
    at the end of every pass."""

    def __init__(self, inv: torch.Tensor):
        self.inv = inv
        self.entry: Optional[tuple] = None
        self.hits = 0

    def quantized(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if SHARE_INPUT:
            try:
                key = (x.data_ptr(), tuple(x.shape), tuple(x.stride()), x.dtype, x.device, x._version)
            except RuntimeError:        # an inference tensor has no version counter: quantise every time
                key = None
            if key is not None and self.entry is not None and self.entry[1] == key:
                self.hits += 1
                return self.entry[2], self.entry[3]
            q, s = int8mm.quantize_rows(x, GROUP, self.inv)
            if key is not None:
                self.entry = (x, key, q, s)
            return q, s
        return int8mm.quantize_rows(x, GROUP, self.inv)

    def clear(self, *_: Any) -> None:
        self.entry = None


# =====================================================================================================================
# The modules
# =====================================================================================================================

def _feats_in(x: Any) -> torch.Tensor:
    return x.feats if hasattr(x, "feats") else x


def _usable(x: torch.Tensor, K: int) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == K and x.numel() > 0
            and not torch.is_autocast_enabled() and not torch.is_grad_enabled() and bool(_state))


def _weight(w: torch.Tensor, how: str, smooth: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The int8 weight and its scales for `how`: per output channel ("rows", "group") or per output channel and
    GROUP input channels ("cg", "cs"), of W * s when there is a smoothing vector s."""
    return int8mm.quantize_weight(w, smooth=smooth, group=GROUP if how in ("cg", "cs") else 0)


def _dequantized(w_q: torch.Tensor, w_s: torch.Tensor, inv: Optional[torch.Tensor], dtype: torch.dtype):
    w = int8mm.dequantize_rows(w_q, w_s)
    if inv is not None:
        w = w * inv[None, :]
    return w.to(dtype)


class Int8Linear(nn.Module):
    """An nn.Linear's work as an int8 GEMM (int8mm.int8_linear): `how` "rows", "group", "cg" or "cs"; `smooth`: the
    SmoothQuant vector s [K] ("cs"; None runs "cs" as "cg"); `shared`: a SharedInput whose 1 / s this layer uses
    (its input then quantised once per pass for every layer that shares it)."""

    def __init__(self, source: nn.Linear, how: str = "rows", smooth: Optional[torch.Tensor] = None,
                 shared: Optional[SharedInput] = None):
        super().__init__()
        w = source.weight.detach()
        N, K = w.shape
        if K % 16 or N % 16 or (how in ("cg", "cs") and K % GROUP):
            raise ValueError(f"int8 GEMMs need K and N in multiples of 16 (K of {GROUP} for weight groups), not "
                             f"{K} x {N}")
        if how not in HOWS:
            raise ValueError(f"how is one of {', '.join(HOWS)}, not {how!r}")
        if smooth is not None and how != "cs":
            raise ValueError("a smoothing vector goes with how = cs")
        if smooth is not None:
            smooth = smooth.to(device=w.device, dtype=torch.float32).contiguous()
        wq, ws = _weight(w, how, smooth)
        self.register_buffer("weight_q", wq)                        # int8 [N, K]
        self.register_buffer("scale", ws)                           # fp32 [N] or [N, K / GROUP]
        self.register_buffer("bias", source.bias.detach().float().contiguous() if source.bias is not None else None)
        if shared is not None:
            self.shared = shared
            self.inv_smooth = shared.inv                            # the shared tensor (not a buffer of each layer)
        else:
            self.shared = None
            self.register_buffer("inv_smooth", (1.0 / smooth) if smooth is not None else None)
        self.in_features, self.out_features, self.how, self.dtype = K, N, how, w.dtype
        self.smoothed = smooth is not None

    def extra_repr(self) -> str:
        what = {"rows": "per-row", "group": "group", "cg": "group x group", "cs": "group x group"}[self.how]
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"int8 {what}{', smoothed' if self.smoothed else ''}{', shared input' if self.shared else ''}")

    def dequantized(self) -> torch.Tensor:
        return _dequantized(self.weight_q, self.scale, self.inv_smooth, self.dtype)

    def forward(self, x: Any) -> Any:
        if hasattr(x, "feats"):
            return x.replace(self._forward(x.feats))
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if not _usable(x, self.in_features):
            bias = self.bias.to(x.dtype) if self.bias is not None else None
            return F.linear(x, self.dequantized(), bias)
        if self.shared is not None:
            y = int8mm.int8_linear(self.shared.quantized(x), self.weight_q, self.scale, self.bias)
            return y.view(*x.shape[:-1], self.out_features)
        return int8mm.int8_linear(x, self.weight_q, self.scale, self.bias,
                                  path="triton" if self.how == "rows" else "group", inv_smooth=self.inv_smooth)


class Int8MLP(nn.Module):
    """A feed-forward's (fc1, GELU tanh, fc2) as int8mm.int8_mlp: fc1 with its GELU in the epilogue, fc2's input
    quantised from it. `how` "rows": per-row scales on both inputs; "group": group scales on both (fc1's input
    quantised in groups, fc1's epilogue writes fc2's input in groups, the hidden layer never in bf16); "cg" / "cs":
    "group" with the weights' scales per output channel and group too, and the smoothing vectors s1 (fc1's input,
    applied as it is quantised) and s2 (fc2's input, applied in fc1's epilogue) where given ("cs"). `how2`: fc2's,
    when it differs ("cg" beside a "cs" fc1)."""

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear, how: str = "rows", how2: Optional[str] = None,
                 smooth1: Optional[torch.Tensor] = None, smooth2: Optional[torch.Tensor] = None):
        super().__init__()
        how2 = how2 or how
        for name, layer, h in (("fc1", fc1, how), ("fc2", fc2, how2)):
            N, K = layer.weight.shape
            if K % 16 or N % 16 or (h != "rows" and N % GROUP) or (h != "rows" and K % GROUP):
                raise ValueError(f"{name}: int8 GEMMs need K and N in multiples of 16 (and of {GROUP} in groups), "
                                 f"not {K} x {N}")
            if h not in HOWS:
                raise ValueError(f"how is one of {', '.join(HOWS)}, not {h!r}")
        if (how == "rows") != (how2 == "rows"):
            raise ValueError("fc1 and fc2 take per-row scales together, or group scales together")
        if (smooth1 is not None and how != "cs") or (smooth2 is not None and how2 != "cs"):
            raise ValueError("a smoothing vector goes with how = cs")
        dev = fc1.weight.device
        s1 = smooth1.to(device=dev, dtype=torch.float32).contiguous() if smooth1 is not None else None
        s2 = smooth2.to(device=dev, dtype=torch.float32).contiguous() if smooth2 is not None else None
        w1q, w1s = _weight(fc1.weight.detach(), how, s1)
        w2q, w2s = _weight(fc2.weight.detach(), how2, s2)
        self.register_buffer("w1_q", w1q)
        self.register_buffer("w1_s", w1s)
        self.register_buffer("b1", fc1.bias.detach().float().contiguous() if fc1.bias is not None else None)
        self.register_buffer("w2_q", w2q)
        self.register_buffer("w2_s", w2s)
        self.register_buffer("b2", fc2.bias.detach().float().contiguous() if fc2.bias is not None else None)
        self.register_buffer("inv_smooth1", (1.0 / s1) if s1 is not None else None)
        self.register_buffer("inv_smooth2", (1.0 / s2) if s2 is not None else None)
        self.in_features, self.hidden, self.out_features = fc1.weight.shape[1], fc1.weight.shape[0], fc2.weight.shape[0]
        self.how, self.how2, self.dtype = how, how2, fc1.weight.dtype

    def extra_repr(self) -> str:
        h = self.how if self.how2 == self.how else f"{self.how} / {self.how2}"
        return f"{self.in_features} -> {self.hidden} -> {self.out_features}, GELU(tanh), int8 {h}"

    def forward(self, x: Any) -> Any:
        if hasattr(x, "feats"):
            return x.replace(self._forward(x.feats))
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.in_features
        if not _usable(x, K):
            w1 = _dequantized(self.w1_q, self.w1_s, self.inv_smooth1, x.dtype)
            w2 = _dequantized(self.w2_q, self.w2_s, self.inv_smooth2, x.dtype)
            h = F.gelu(F.linear(x, w1, self.b1.to(x.dtype) if self.b1 is not None else None), approximate="tanh")
            return F.linear(h, w2, self.b2.to(x.dtype) if self.b2 is not None else None)
        flat = x.reshape(-1, K)
        if flat.stride(-1) != 1:
            flat = flat.contiguous()
        w1, w2 = (self.w1_q, self.w1_s, self.b1), (self.w2_q, self.w2_s, self.b2)
        if self.how == "rows":
            y = int8mm.int8_mlp(flat, w1, w2, path="rows")
        elif self.inv_smooth1 is None and self.inv_smooth2 is None:
            y = int8mm.int8_mlp(int8mm.quantize_rows(flat, GROUP), w1, w2, path="group")
        else:
            y = int8mm.int8_mlp(int8mm.quantize_rows(flat, GROUP, self.inv_smooth1), w1, w2, path="group",
                                inv_smooth2=self.inv_smooth2)
        return y.view(*x.shape[:-1], self.out_features)


# =====================================================================================================================
# The conversion
# =====================================================================================================================

_BLOCK = re.compile(r"^blocks\.(\d+)\.")


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


def _resolve(rules: dict[str, str], calib: Optional[dict[str, torch.Tensor]]) -> dict[str, str]:
    """The rules a flow runs: "cs" without a calibration file is "cg", and proj_linear is then left in bf16."""
    if calib is not None or "cs" not in rules.values():
        return dict(rules)
    out = {t: ("cg" if h == "cs" else h) for t, h in rules.items()}
    if rules.get("proj") == "cs":
        del out["proj"]
    return out


def convert(model: nn.Module, kind: str = "", checkpoint: str = "") -> nn.Module:
    """The model (a flow of `kind`: ss, lr, hr or tex, loaded from `checkpoint`: the checkpoint's file name without
    extension, which names its calibration file) with its token-sized linears as int8 layers where the policy says
    so, when the switch is on and the card and the self-check allow it; the model unchanged otherwise. Called by the
    staged loader after a flow network is loaded (its weights on the card in bf16; each bf16 weight is freed as it
    is replaced)."""
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
    calib = load_calib(checkpoint) if "cs" in rules.values() else None
    effective = _resolve(rules, calib)
    if "cs" in rules.values() and calib is None:
        print(f"[INT8] no SmoothQuant calibration for {Path(str(checkpoint)).name or '(no checkpoint name)'} "
              f"({calib_file(checkpoint or '?').name} in {CALIB_DIR}): {kind} runs policy {name} without the "
              f"smoothing (group scales only) and keeps proj_linear in bf16", flush=True)
    before = _bytes(model)
    counts: dict[str, int] = {}
    smoothed = 0
    shared: dict[str, SharedInput] = {}

    def vector(layer_name: str, layer_type: str, how: str) -> tuple[Optional[torch.Tensor], Optional[SharedInput]]:
        """(s, None) for a per-layer vector, (None, SharedInput) for a shared one, (None, None) without."""
        if how != "cs" or calib is None:
            return None, None
        if layer_name in calib:
            return calib[layer_name].to(device), None
        key = f"shared.{layer_type}"
        if key in calib:
            if layer_type not in shared:
                shared[layer_type] = SharedInput((1.0 / calib[key].to(device)).contiguous())
            return calib[key].to(device), shared[layer_type]
        return None, None

    with torch.no_grad():
        for mod_name, module in list(model.named_modules()):
            hit = _BLOCK.match(mod_name + ".")
            prefix = f"blocks.{hit.group(1)}." if hit else ""
            mlp = getattr(module, "mlp", None)
            if (type(module).__name__ in _FFN and "fc1" in effective and "fc2" in effective
                    and isinstance(mlp, nn.Sequential) and len(mlp) == 3 and isinstance(mlp[0], nn.Linear)
                    and isinstance(mlp[2], nn.Linear) and type(mlp[1]).__name__ in ("SparseGELU", "GELU")):
                h1, h2 = effective["fc1"], effective["fc2"]
                s1, _ = vector(prefix + PATHS["fc1"], "fc1", h1)
                s2, _ = vector(prefix + PATHS["fc2"], "fc2", h2)
                if (h1 == "cs" and s1 is None) or (h2 == "cs" and s2 is None):
                    raise RuntimeError(f"the calibration of {checkpoint} has no vector for {prefix}mlp")
                module.mlp = Int8MLP(mlp[0], mlp[2], h1, h2, smooth1=s1, smooth2=s2)
                smoothed += (s1 is not None) + (s2 is not None)
                counts["mlp"] = counts.get("mlp", 0) + 1
                del mlp
                continue
            for layer, container, key, how in _targets(module, effective):
                old = getattr(container, key)
                full = f"{mod_name}.{key}" if mod_name else key
                s, share = vector(full, layer, how)
                if how == "cs" and s is None:
                    raise RuntimeError(f"the calibration of {checkpoint} has no vector for {full}")
                setattr(container, key, Int8Linear(old, how, smooth=s if share is None else s, shared=share))
                smoothed += s is not None
                counts[layer] = counts.get(layer, 0) + 1
                del old
    for sh in shared.values():
        model.register_forward_pre_hook(sh.clear)
        model.register_forward_hook(sh.clear)
    if counts:
        print(f"[INT8] {type(model).__name__} ({kind}, policy {name}): "
              + ", ".join(f"{k} x{v}" for k, v in counts.items())
              + f" as int8 GEMMs ({KERNEL}; PIXAL3D_INT8)"
              + (f", {smoothed} smoothed ({calib_file(checkpoint).name})" if smoothed else "")
              + f", {before / 2**20:,.0f} -> {_bytes(model) / 2**20:,.0f} MiB of weights", flush=True)
    return model


def _bytes(model: nn.Module) -> int:
    seen: set = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            total += t.numel() * t.element_size()
    return total


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
