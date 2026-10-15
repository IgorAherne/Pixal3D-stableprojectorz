"""SageAttention (int8 Q.K^T, fp16 P.V) for the flows' full self-attention, behind the switch PIXAL3D_SAGE (on by default).

The flows' self-attention over all tokens of one prop (sparse: lr, hr, tex - one sequence of N tokens; dense: ss -
4096 tokens) is ~60 % of a detail pass at 1536, and flash_attn already runs it at the card's bf16 tensor peak.
SageAttention (thu-ml, Windows wheels by woct0rdho) quantises q and k to int8 (one scale per block of tokens and
head, k's per-channel mean over the tokens subtracted first) and multiplies P by v in fp16: about twice flash_attn's
speed on an RTX 3060 (an hr / tex pass at 1536: -26 %; ss -7 %, lr -3 %), at a cost in accuracy that is measured,
not free (code/devtools/pixal3d-lowmem/round-g/sage in Spellcast3D: on the flows' real q, k, v at 1536 its error
against an fp32 reference is 1.15-10.4 times flash_attn's own - which is the bf16 rounding of the output - and up to
~30 times in the heads whose k has a few dominant channels; a whole hr sampling ends ~1.2 times as far from the
flash_attn run as the fused-vs-eager change does).

The switch is the environment variable PIXAL3D_SAGE, read when the attention config modules are imported
(modules/sparse/config.py and modules/attention/config.py hold it as `SAGE`; their set_sage() changes it in a
running process):

    "on", "1", "true", "yes"     SageAttention where it applies (below), after its self-check
    "off", "0", "false", "no"    flash_attn everywhere, as before: the same numbers bit for bit
    unset, ""                    DEFAULT (below)
    anything else                off, with a warning

Where it applies: CUDA tensors in bf16, no autograd, no autocast, head size 128, the full self-attention of ONE
sequence (sparse: q, k, v sparse tensors of one prop with the same length, the call with three arguments; dense:
q, k, v [B, L, H, 128] with as many keys as queries). Cross-attention (5 context tokens), several sequences and
every other call keep their backend.

The kernel: on sm_8x (RTX 30xx / 40xx, A100) sageattn_qk_int8_pv_fp16_cuda with qk_quant_gran "per_warp" and
pv_accum_dtype "fp16+fp32" (the SASS for sm_80 runs on every 8.x card); on sm_90 and newer, whose cards cannot run
that SASS, sageattn_qk_int8_pv_fp16_triton (the same quantisation and accumulation, compiled by Triton). Both sum
P.V over 64 keys in fp16 before adding to fp32, and cast v to fp16: v is therefore scaled by 2^-V_SHIFT before the
call and the result by 2^V_SHIFT after it - exact in bf16, the same values to the last digit (measured) - so that
neither overflows below |v| ~ 262,000 (the flows' largest |v| measured: 860; without the scaling the fp16 partial
sums could overflow from |v| ~ 1,023, and the pure-fp16 accumulator, "fp16", gives Inf on the flows' real tensors).
Sequences of any length are handled by the kernels (masked last block; lengths of 64 tokens or fewer are not
taken: the wheel's sm_80 kernel reads unset shared memory there, fixed upstream after the build we use).

Before the first SageAttention call of a process the kernel runs once on a small random table against an fp32
reference (`_check`); a card older than sm_80, a missing or broken wheel, or a kernel that fails or answers wrongly
turns it off with a `[SAGE]` warning, and the flows keep flash_attn.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

DEFAULT = True      # what an unset PIXAL3D_SAGE means: on, after the self-check (round G: the five test props at 1536
#                     made with it are of the same quality as with flash_attn - Chamfer 0.17-0.24 % of the bounding-box
#                     diagonal, the class of the fused kernels' own change - for -20 % of a batch)
V_SHIFT = 8         # v * 2^-8 into the kernel, its answer * 2^8: exact in bf16
HEAD_DIM = 128      # the only head size checked
MIN_TOKENS = 65     # one block of 64 keys or fewer: see the docstring

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}


def from_env() -> bool:
    """PIXAL3D_SAGE as a bool (the config modules call this once, when they are imported)."""
    text = os.environ.get("PIXAL3D_SAGE", "").strip().lower()
    if text == "":
        return DEFAULT
    if text in _ON:
        return True
    if text not in _OFF and text not in _warned:
        _warned.add(text)
        print(f"[SAGE] PIXAL3D_SAGE={text!r} is neither on nor off: SageAttention stays off", flush=True)
    return False


_warned: set = set()


_state: Optional[bool] = None       # None: not tried yet; True: passed its self-check on this card; False: off
_kernel: Optional[Callable] = None
KERNEL = ""                         # the kernel in use, for reports


def usable(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether this self-attention may run SageAttention (the caller has checked the switch and that it is the full
    self-attention of one sequence). The first time, the kernel is tried on the card (`_check`)."""
    if (not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or torch.is_grad_enabled() or torch.is_autocast_enabled()):
        return False
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        return False
    if q.shape[-3] < MIN_TOKENS or k.shape[-3] < MIN_TOKENS:
        return False
    if _state is None:
        _check(q.device)
    return bool(_state)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Softmax(q k^T / sqrt(D)) v for q, k, v [B, N, H, D] (last dimension contiguous), not causal: [B, N, H, D] in
    q's dtype. Call only where usable() said True."""
    out = _kernel(q, k, v * 2.0 ** -V_SHIFT)
    return out.mul_(2.0 ** V_SHIFT)


def _choose(device: torch.device) -> tuple[Callable, str]:
    import sageattention as sa

    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError(f"the card is sm_{major}{minor}; SageAttention's int8 kernels are used from sm_80 on")
    if major == 8:
        def run(q, k, v):
            return sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD", is_causal=False,
                                                    qk_quant_gran="per_warp", pv_accum_dtype="fp16+fp32")
        return run, "sageattn_qk_int8_pv_fp16_cuda (per_warp, fp16+fp32)"

    def run(q, k, v):
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD", is_causal=False,
                                                  quantization_backend="triton")
    return run, "sageattn_qk_int8_pv_fp16_triton"


def _check(device: torch.device) -> bool:
    """The kernel once on a random table (1000 tokens: more than one block of keys and not a multiple of it; v as
    a strided view and 1000 times larger than q and k, as the flows' v is), against fp32 attention: finite, and
    within 3 % relative L2 (it measures ~1.3 % there, flash_attn ~0.23 %). Anything else turns it off."""
    global _state, _kernel, KERNEL
    try:
        _kernel, KERNEL = _choose(device)
        g = torch.Generator(device=device).manual_seed(0)
        n, heads = 1000, 12
        qkv = torch.randn((1, n, 3, heads, HEAD_DIM), device=device, dtype=torch.bfloat16, generator=g)
        qkv[:, :, 2] *= 1000
        q, k, v = qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2]
        got = attention(q, k, v).float()
        want = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2),
                                              v.float().transpose(1, 2)).transpose(1, 2)
        torch.cuda.synchronize(device)
        rel = float((got - want).norm() / want.norm())
        if got.shape != want.shape or not bool(torch.isfinite(got).all()) or not rel <= 0.03:
            raise RuntimeError(f"its answer differs from fp32 attention (relative L2 {rel:.3g}, shape "
                               f"{tuple(got.shape)})")
    except Exception as exc:        # noqa: BLE001 - whatever the wheel, the card or Triton raises: flash_attn serves
        _state, _kernel = False, None
        print(f"[SAGE] SageAttention does not run on this card ({type(exc).__name__}: {str(exc)[:300]}); the "
              f"flows keep flash_attn", flush=True)
        return False
    _state = True
    print(f"[SAGE] the flows' full self-attention runs SageAttention: {KERNEL} (PIXAL3D_SAGE)", flush=True)
    return True
