"""The int8 (W8A8) kernels of the flows' linear layers (modules/int8.py chooses where they run): Triton GEMMs

    y = (x_int8 @ w_int8^T) [int32 accumulate] * row_scale[M] * col_scale[N] + bias   [-> GELU(tanh)]  -> bf16

Symmetric quantisation, no zero points: weights per output channel (one-off at load), activations per row (token)
at run time, or per row and per 128 input channels ("group" scales, a finer grain for the quality study).

    quantize_weight(w)            bf16 [N, K] -> (int8 [N, K] row-major, fp32 col_scales [N])
    quantize_rows(x, group=0)     bf16 [..., K] -> (int8 [M, K], fp32 scales [M] or [M, K / group])
    norm_quant(x, eps, ...)       fused.norm_modulate's math (LayerNorm32 [affine] [* (1 + scale) + shift], the
                                  default "rounded" mode) emitting int8 rows + scales instead of bf16
    int8_linear(x, w_q, w_s, bias, *, gelu=False, path=None, out_int8=False)
                                  x: bf16 [..., K] or a pre-quantised (x_q, x_s) pair -> bf16 [..., N]
                                  (out_int8: the result requantised to int8 + scales, the next layer's input)
    int8_mlp(x, w1, w2)           fc1 + GELU -> int8 (group scales) -> fc2: the [M, 8192] hidden layer never in bf16

Paths (`path=`; None = the table PATHS, chosen from the round-I measurements on the RTX 3060):
    "triton"   quantize_rows + one Triton GEMM with the scaling / bias / GELU epilogue fused (per-row scales).
               THE FASTEST at every measured shape: 2.5-3.2x the bf16 F.linear with the quantiser in the call,
               2.9-3.6x when the input arrives as int8 (norm_quant, fc1's epilogue); 84-88 TOPS.
    "intmm"    quantize_rows + torch._int_mm (cuBLASLt IMMA, int32 [M, N] out, rows in chunks of CHUNK_ROWS)
               + one Triton epilogue kernel: 1.6-2.7x (the int32 round trip); ties "triton" only on fc2 at small M.
    "group"    quantize_rows(group=128) + the Triton GEMM with the accumulator rescaled every 128-K block:
               1.2-1.3x slower than "triton" (66-68 TOPS), for the quality study's finer scales.
The whole block (3 norms + 7 token-sized linears + GELU) runs 2.67-2.83x faster than bf16 (README.md).

Capability gate (`capability()`): int8 tensor-core MMA exists from sm_75 (Turing). torch._int_mm needs sm_75+ and
cuBLASLt; the Triton kernel needs Triton's int8 mma.sync (m16n8k32), which Triton emits for sm_80+ (Ampere and up);
only sm_86 (RTX 3060) was measured. Below sm_75, or without Triton for the Triton paths, or when the first-use
self-check (`self_check`) fails, every call runs F.linear on the dequantised weight (the fallback) and says so once.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

GROUP = 128                 # activation group size of the "group" path (= that kernel's BLOCK_K)
CHUNK_ROWS = 8192           # "intmm": rows per torch._int_mm call (bounds the int32 temporary: 8192 x N x 4 B)
TINY = 1e-30                # absmax floor: an all-zero row quantises to zeros with a tiny scale, not to NaN
NORM_WARPS = 4              # norm_quant's num_warps (C = 1536): 4 warps 0.461 ms, 8 warps 0.559 ms at M = 21504

# Triton GEMM configurations: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps), from the sweep of
# 14 configs x 6 shapes x M in {2024, 4096, 8192, 21504, 30000} on the RTX 3060 (bench_int8.py --parts triton):
CFG_BIG = (128, 128, 64, 8, 4, 4)       # the fastest for every K <= 2048 shape at every M (84-88 TOPS)
CFG_LONGK = (128, 256, 64, 8, 3, 8)     # K >= 4096 (fc2): CFG_BIG falls to 50 TOPS there (A re-read past the L2)
CFG_GROUP = (128, 128, 128, 8, 3, 8)    # group scales: BLOCK_K = 128 = the group; 4-warp / 256-wide tiles spill
CFG_FC1_INT8 = (128, 128, 64, 8, 4, 4)  # fc1 + GELU -> int8 tiles (BLOCK_N = GROUP): spills 144 B and is still
#                                         the faster (MLP 15.7 ms vs 18.1 ms with the 8-warp tile at M = 21504)

# per (N, K): the path; measured fastest everywhere: the Triton GEMM with the fused epilogue (README.md)
PATHS: dict = {}
DEFAULT_PATH = "triton"

_kernels = None
_state = {"ok": None, "why": "", "warned": False}


# =====================================================================================================================
# Capability gate
# =====================================================================================================================

def capability(device: Optional[torch.device] = None) -> Tuple[bool, bool, str]:
    """(int8 tensor cores usable at all [torch._int_mm], Triton int8 GEMM usable, the reason when not)."""
    if not torch.cuda.is_available():
        return False, False, "no CUDA device"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (7, 5):
        return False, False, f"sm_{major}{minor}: no int8 tensor-core MMA below sm_75"
    try:
        import triton  # noqa: F401
    except Exception as exc:        # noqa: BLE001
        return True, False, f"Triton not importable ({type(exc).__name__})"
    if major < 8:
        return True, False, f"sm_{major}{minor}: Triton's int8 mma.sync path needs sm_80+; torch._int_mm only"
    return True, True, ""


# =====================================================================================================================
# Kernels
# =====================================================================================================================

def _build():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def round_bf16(x):
        # as modules/fused.py: x rounded to the nearest bf16 (ties to even) by integer arithmetic on the bits
        bits = x.to(tl.uint32, bitcast=True)
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
        return tl.where(x != x, x, bits.to(tl.float32, bitcast=True))

    @triton.jit
    def quant_row(y, Q, S, row, cols, mask, stride_q):
        # one fp32 row (masked lanes 0) -> int8 + its scale; torch.round(y * (127 / amax)) bit for bit
        amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
        inv = tl.div_rn(127.0, amax)
        q = tl.minimum(tl.maximum(libdevice.rint(y * inv), -127.0), 127.0)
        tl.store(Q + row * stride_q + cols, q.to(tl.int8), mask=mask)
        tl.store(S + row, tl.div_rn(amax, 127.0))

    @triton.jit(do_not_specialize=["M", "stride_s"])
    def quant_rows_kernel(X, Q, S, SM, M, K, stride_x, stride_q, stride_s,
                          GROUP: tl.constexpr, NG: tl.constexpr, BLOCK_K: tl.constexpr, HAS_SMOOTH: tl.constexpr):
        # HAS_SMOOTH: x * SM[K], SM = 1 / s (SmoothQuant's per-input-channel divisor s; the weight holds W * s)
        row = tl.program_id(0).to(tl.int64)
        if GROUP == 0:
            cols = tl.arange(0, BLOCK_K)
            mask = cols < K
            x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_SMOOTH:
                x = x * tl.load(SM + cols, mask=mask, other=1.0).to(tl.float32)
            quant_row(x, Q, S, row, cols, mask, stride_q)
        else:
            g = tl.arange(0, NG)
            c = tl.arange(0, GROUP)
            cols = g[:, None] * GROUP + c[None, :]
            gmask = g < K // GROUP
            x = tl.load(X + row * stride_x + cols, mask=gmask[:, None], other=0.0).to(tl.float32)
            if HAS_SMOOTH:
                x = x * tl.load(SM + cols, mask=gmask[:, None], other=1.0).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
            inv = tl.div_rn(tl.full([NG], 127.0, tl.float32), amax)
            q = tl.minimum(tl.maximum(libdevice.rint(x * inv[:, None]), -127.0), 127.0)
            tl.store(Q + row * stride_q + cols, q.to(tl.int8), mask=gmask[:, None])
            tl.store(S + row * stride_s + g, tl.div_rn(amax, tl.full([NG], 127.0, tl.float32)), mask=gmask)

    @triton.jit(do_not_specialize=["n_rows", "rows_per_batch", "stride_s"])
    def norm_quant_kernel(X, Q, S, W, B, SCALE, SHIFT, BIDX, SM, n_rows, rows_per_batch, C, stride_x, stride_q,
                          stride_s, stride_mod, eps, AFFINE: tl.constexpr, MODULATE: tl.constexpr,
                          HAS_BIDX: tl.constexpr, ROUNDED: tl.constexpr, GROUP: tl.constexpr, NG: tl.constexpr,
                          BLOCK_C: tl.constexpr, HAS_SMOOTH: tl.constexpr):
        # modules/fused.py norm_modulate_kernel's arithmetic, the result rounded to bf16 (the value the bf16 path
        # hands to the linear), then quantised: equal to quantize_rows(norm_modulate(x)) bit for bit
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / C
        rstd = tl.math.rsqrt(var + eps)
        y = xc * rstd
        if AFFINE:
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            y = w * y + b
        if MODULATE:
            if HAS_BIDX:
                bi = tl.load(BIDX + row).to(tl.int64)
            else:
                bi = row // rows_per_batch
            scale = tl.load(SCALE + bi * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(SHIFT + bi * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
            if ROUNDED:
                y = round_bf16(y)
                s1 = round_bf16(scale + 1.0)
                y = round_bf16(y * s1)
                y = y + shift
            else:
                y = y * (scale + 1.0) + shift
        y = tl.where(mask, round_bf16(y), 0.0)          # the bf16 value the eager / fused path would store
        if HAS_SMOOTH:
            y = y * tl.load(SM + cols, mask=mask, other=1.0).to(tl.float32)
        if GROUP == 0:
            quant_row(y, Q, S, row, cols, mask, stride_q)
        else:
            # [BLOCK_C] -> [NG, GROUP]: one scale per GROUP channels
            y2 = tl.reshape(y, (NG, GROUP))
            g = tl.arange(0, NG)
            c2 = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
            gmask = g < C // GROUP
            amax = tl.maximum(tl.max(tl.abs(y2), axis=1), 1e-30)
            inv = tl.div_rn(tl.full([NG], 127.0, tl.float32), amax)
            q = tl.minimum(tl.maximum(libdevice.rint(y2 * inv[:, None]), -127.0), 127.0)
            tl.store(Q + row * stride_q + c2, q.to(tl.int8), mask=gmask[:, None])
            tl.store(S + row * stride_s + g, tl.div_rn(amax, tl.full([NG], 127.0, tl.float32)), mask=gmask)

    @triton.jit(do_not_specialize=["M", "stride_asm", "stride_qsm"])
    def gemm_kernel(A, B, C, AS, BS, BIAS, QS, M, N, K, stride_am, stride_bn, stride_cm, stride_asm, stride_qsm,
                    HAS_BIAS: tl.constexpr, GELU: tl.constexpr, OUT: tl.constexpr, A_GROUP: tl.constexpr,
                    MAGIC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    GROUP_M: tl.constexpr):
        # A: int8 [M, K] row-major; B: int8 [N, K] row-major (the weight as stored: both operands K-contiguous,
        # the "TN" layout int8 mma wants). OUT: 0 = bf16, 1 = the raw int32 accumulator (checks), 2 = int8 with one
        # fp32 scale per row and per BLOCK_N columns (QS [M, N / BLOCK_N]: the next layer's group-scaled input).
        # A_GROUP: 0 = AS holds one scale per row; else AS [M, K / BLOCK_K] and the int32 partial product of each
        # BLOCK_K slab is scaled into an fp32 accumulator (MAGIC: int32 -> fp32 by the 1.5 * 2^23 trick, exact for
        # |partial| < 2^22, i.e. BLOCK_K <= 256).
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a_ptrs = A + ram[:, None].to(tl.int64) * stride_am + rk[None, :]
        b_ptrs = B + rbn[None, :].to(tl.int64) * stride_bn + rk[:, None]
        if A_GROUP:
            accf = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            s_ptrs = AS + ram.to(tl.int64) * stride_asm
            for kb in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                part = tl.dot(a, b, out_dtype=tl.int32)
                s = tl.load(s_ptrs + kb)
                if MAGIC:
                    pf = (part + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0
                else:
                    pf = part.to(tl.float32)
                accf += pf * s[:, None]
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K
            y = accf
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kb in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K
            rs = tl.load(AS + ram)
            y = acc.to(tl.float32) * rs[:, None]
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        c_ptrs = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :]
        if OUT == 1:
            tl.store(c_ptrs, acc, mask=mask)
        else:
            cs = tl.load(BS + rbn)
            y = y * cs[None, :]
            if HAS_BIAS:
                y += tl.load(BIAS + rbn).to(tl.float32)[None, :]
            if GELU:
                inner = 0.7978845608028654 * (y + 0.044715 * (y * y * y))
                y = 0.5 * y * (1.0 + libdevice.tanh(inner))
            if OUT == 2:
                amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-30)
                inv = tl.div_rn(tl.full([BLOCK_M], 127.0, tl.float32), amax)
                q = tl.minimum(tl.maximum(libdevice.rint(y * inv[:, None]), -127.0), 127.0)
                tl.store(c_ptrs, q.to(tl.int8), mask=mask)
                tl.store(QS + rm.to(tl.int64) * stride_qsm + pid_n,
                         tl.div_rn(amax, tl.full([BLOCK_M], 127.0, tl.float32)), mask=rm < M)
            else:
                tl.store(c_ptrs, y.to(tl.bfloat16), mask=mask)

    @triton.jit(do_not_specialize=["M"])
    def epilogue_kernel(C32, Y, AS, BS, BIAS, QS, M, N, stride_c, stride_y, HAS_BIAS: tl.constexpr,
                        GELU: tl.constexpr, OUT: tl.constexpr, BLOCK_N: tl.constexpr):
        # torch._int_mm's int32 [M, N] -> * row_scale * col_scale + bias [-> GELU] -> bf16 (OUT 0) or int8 with one
        # scale per row (OUT 2: the input of a following torch._int_mm). One row per program.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        acc = tl.load(C32 + row * stride_c + cols, mask=mask, other=0)
        rs = tl.load(AS + row)
        cs = tl.load(BS + cols, mask=mask, other=0.0)
        y = (acc.to(tl.float32) * rs) * cs
        if HAS_BIAS:
            y += tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
        if GELU:
            inner = 0.7978845608028654 * (y + 0.044715 * (y * y * y))
            y = 0.5 * y * (1.0 + libdevice.tanh(inner))
        if OUT == 2:
            quant_row(y, Y, QS, row, cols, mask, stride_y)
        else:
            tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)

    _kernels = {"quant": quant_rows_kernel, "norm_quant": norm_quant_kernel, "gemm": gemm_kernel,
                "epilogue": epilogue_kernel, "next_pow2": triton.next_power_of_2, "cdiv": triton.cdiv}
    return _kernels


def _p2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _warps(width: int) -> int:
    return 4 if width <= 2048 else 8


# =====================================================================================================================
# Quantisers
# =====================================================================================================================

def quantize_rows(x: torch.Tensor, group: int = 0,
                  inv_smooth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 (or fp16 / fp32) [..., K] -> (int8 [M, K], fp32 scales [M] or [M, K // group]); symmetric,
    q = round_half_even(x * (127 / absmax)), scale = absmax / 127 (absmax floored at TINY). inv_smooth: 1 / s
    (fp32 [K], computed once at load) for SmoothQuant's per-input-channel vector s, applied as x * (1 / s) in the
    same pass (the weight was quantised as W * s: quantize_weight(w, smooth=s))."""
    k = _build()
    K = x.shape[-1]
    flat = x.reshape(-1, K)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    M = flat.shape[0]
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    if group:
        if K % group or group & (group - 1):
            raise ValueError(f"group {group} must be a power of two dividing K = {K}")
        s = torch.empty((M, K // group), dtype=torch.float32, device=x.device)
        ng, block, stride_s = _p2(K // group), 1, s.stride(0)
    else:
        s = torch.empty((M,), dtype=torch.float32, device=x.device)
        ng, block, stride_s = 1, _p2(K), 1
    if M:
        k["quant"][(M,)](flat, q, s, inv_smooth if inv_smooth is not None else flat, M, K, flat.stride(0),
                         q.stride(0), stride_s, GROUP=group, NG=ng, BLOCK_K=block, HAS_SMOOTH=inv_smooth is not None,
                         num_warps=_warps(K))
    return q, s


def quantize_weight(w: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """nn.Linear weight [N, K] -> (int8 [N, K] row-major, fp32 per-output-channel scales [N]). The same row
    quantiser: a row of the weight is one output channel. One-off, at load. smooth: s [K]; W * s is quantised
    (the activations then pass inv_smooth = 1 / s to quantize_rows / norm_quant / int8_linear)."""
    w = w.detach()
    if smooth is not None:
        w = w.float() * smooth.float()[None, :]
    return quantize_rows(w, 0)


def quantize_rows_torch(x: torch.Tensor, group: int = 0,
                        inv_smooth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The reference quantiser in plain torch ops (the Triton kernels must equal it bit for bit). Note: torch's
    `127.0 / t` is t.reciprocal() * 127 (two roundings); the kernels divide once (IEEE), as torch.div of two
    tensors does."""
    K = x.shape[-1]
    xf = x.reshape(-1, K).float()
    if inv_smooth is not None:
        xf = xf * inv_smooth.float()
    if group:
        xf = xf.view(xf.shape[0], K // group, group)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(TINY)
    q = torch.round(xf * torch.div(torch.full_like(amax, 127.0), amax)).clamp_(-127, 127).to(torch.int8)
    s = torch.div(amax, torch.full_like(amax, 127.0)).squeeze(-1)   # IEEE (t / 127.0 is t * (1 / 127))
    return q.reshape(-1, K), s


def norm_quant(x: torch.Tensor, eps: float, *, weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None,
               shift: Optional[torch.Tensor] = None, batch: int = 1, rows_per_batch: Optional[int] = None,
               batch_index: Optional[torch.Tensor] = None, group: int = 0, rounded: bool = True,
               inv_smooth: Optional[torch.Tensor] = None, num_warps: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """fused.norm_modulate(x, ...) followed by quantize_rows, in one kernel: (int8 [M, C], scales)."""
    k = _build()
    C = x.shape[-1]
    flat = x.reshape(-1, C)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    n_rows = flat.shape[0]

    def mod_rows(mod):
        if mod is None:
            return None, 0
        if mod.dim() != 2:
            mod = mod.reshape(mod.shape[0], -1)
        return mod, (0 if mod.shape[0] == 1 else mod.stride(0))

    scale, stride_mod = mod_rows(scale)
    shift, _ = mod_rows(shift)
    q = torch.empty((n_rows, C), dtype=torch.int8, device=x.device)
    if group:
        if C % group:
            raise ValueError(f"group {group} must divide C = {C}")
        s = torch.empty((n_rows, C // group), dtype=torch.float32, device=x.device)
        block = max(_p2(C), group)
        ng, stride_s = block // group, s.stride(0)
    else:
        s = torch.empty((n_rows,), dtype=torch.float32, device=x.device)
        block, ng, stride_s = _p2(C), 1, 1
    if n_rows:
        k["norm_quant"][(n_rows,)](
            flat, q, s, weight if weight is not None else flat, bias if bias is not None else flat,
            scale if scale is not None else flat, shift if shift is not None else flat,
            batch_index if batch_index is not None else flat, inv_smooth if inv_smooth is not None else flat, n_rows,
            rows_per_batch or n_rows, C, flat.stride(0), q.stride(0), stride_s, stride_mod, float(eps),
            AFFINE=weight is not None, MODULATE=scale is not None, HAS_BIDX=batch_index is not None,
            ROUNDED=rounded, GROUP=group, NG=ng, BLOCK_C=block, HAS_SMOOTH=inv_smooth is not None,
            num_warps=num_warps or NORM_WARPS)
    return q, s


# =====================================================================================================================
# GEMMs
# =====================================================================================================================

def pick_config(M: int, N: int, K: int, group: int = 0) -> tuple:
    if group:
        return CFG_GROUP
    return CFG_LONGK if K >= 4096 else CFG_BIG


def gemm(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor],
         *, gelu: bool = False, out: str = "bf16", group: int = 0, cfg: Optional[tuple] = None,
         magic: bool = True):
    """The Triton int8 GEMM with the epilogue fused. out: "bf16" -> bf16 [M, N]; "int32" -> the raw accumulator
    (per-row mode only); "int8" -> (int8 [M, N], fp32 scales [M, N // BLOCK_N])."""
    k = _build()
    M, K = xq.shape
    N = w_q.shape[0]
    BM, BN, BK, GM, stages, warps = cfg or pick_config(M, N, K, group)
    if K % BK:
        raise ValueError(f"K = {K} is not a multiple of BLOCK_K = {BK}")
    if group and group != BK:
        raise ValueError(f"the group path needs BLOCK_K == group ({BK} != {group})")
    mode = {"bf16": 0, "int32": 1, "int8": 2}[out]
    dtype = {0: torch.bfloat16, 1: torch.int32, 2: torch.int8}[mode]
    y = torch.empty((M, N), dtype=dtype, device=xq.device)
    qs = None
    if mode == 2:
        if N % BN:
            raise ValueError(f"int8 output needs N % BLOCK_N == 0 ({N} % {BN})")
        qs = torch.empty((M, N // BN), dtype=torch.float32, device=xq.device)
    if M:
        grid = (k["cdiv"](M, BM) * k["cdiv"](N, BN),)
        k["gemm"][grid](xq, w_q, y, xs, w_s, bias if bias is not None else w_s, qs if qs is not None else w_s,
                        M, N, K, xq.stride(0), w_q.stride(0), y.stride(0), xs.stride(0) if group else 1,
                        qs.stride(0) if qs is not None else 1, HAS_BIAS=bias is not None, GELU=gelu, OUT=mode,
                        A_GROUP=group, MAGIC=magic, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                        num_stages=stages, num_warps=warps)
    return (y, qs) if mode == 2 else y


def epilogue(c32: torch.Tensor, xs: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor], *,
             gelu: bool = False, out_int8: bool = False, y: Optional[torch.Tensor] = None,
             ys: Optional[torch.Tensor] = None):
    """int32 [M, N] -> bf16 [M, N] (or int8 [M, N] + per-row scales [M]); y / ys: optional outputs to write."""
    k = _build()
    M, N = c32.shape
    if y is None:
        y = torch.empty((M, N), dtype=torch.int8 if out_int8 else torch.bfloat16, device=c32.device)
    if out_int8 and ys is None:
        ys = torch.empty((M,), dtype=torch.float32, device=c32.device)
    if M:
        k["epilogue"][(M,)](c32, y, xs, w_s, bias if bias is not None else w_s, ys if out_int8 else w_s, M, N,
                            c32.stride(0), y.stride(0), HAS_BIAS=bias is not None, GELU=gelu,
                            OUT=2 if out_int8 else 0, BLOCK_N=_p2(N), num_warps=_warps(N))
    return (y, ys) if out_int8 else y


def intmm_linear(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                 bias: Optional[torch.Tensor], *, gelu: bool = False, out_int8: bool = False,
                 chunk: Optional[int] = CHUNK_ROWS):
    """torch._int_mm (rows in chunks: the int32 temporary is chunk x N x 4 B) + the Triton epilogue per chunk."""
    M, K = xq.shape
    N = w_q.shape[0]
    if M <= 16:                                     # torch._int_mm wants more than 16 rows
        pad = torch.zeros((17, K), dtype=xq.dtype, device=xq.device)
        pad[:M] = xq
        ps = torch.zeros((17,), dtype=xs.dtype, device=xs.device)
        ps[:M] = xs
        r = intmm_linear(pad, ps, w_q, w_s, bias, gelu=gelu, out_int8=out_int8, chunk=None)
        return (r[0][:M], r[1][:M]) if out_int8 else r[:M]
    wt = w_q.t()                                    # [K, N] column-major: the layout cuBLASLt's IMMA kernels take
    y = torch.empty((M, N), dtype=torch.int8 if out_int8 else torch.bfloat16, device=xq.device)
    ys = torch.empty((M,), dtype=torch.float32, device=xq.device) if out_int8 else None
    step = M if not chunk or chunk >= M else chunk
    buf = torch.empty((step, N), dtype=torch.int32, device=xq.device)
    for r0 in range(0, M, step):
        r1 = min(M, r0 + step)
        if r1 - r0 <= 16:                           # a last chunk of <= 16 rows: redo the 17 last rows instead
            r0 = r1 - 17
        c = buf[: r1 - r0]
        torch._int_mm(xq[r0:r1], wt, out=c)
        epilogue(c, xs[r0:r1], w_s, bias, gelu=gelu, out_int8=out_int8, y=y[r0:r1],
                 ys=ys[r0:r1] if ys is not None else None)
    return (y, ys) if out_int8 else y


def dequantize_rows(xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """(int8 [M, K], scales [M] or [M, K / group]) -> fp32 [M, K]."""
    M, K = xq.shape
    if xs.dim() == 2:
        return (xq.float().view(M, xs.shape[1], -1) * xs.view(M, -1, 1)).view(M, K)
    return xq.float() * xs[:, None]


def _fallback(x2: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor],
              gelu: bool) -> torch.Tensor:
    w = (w_q.float() * w_s[:, None]).to(x2.dtype if x2.dtype.is_floating_point else torch.bfloat16)
    y = F.linear(x2.to(w.dtype), w, bias.to(w.dtype) if bias is not None else None)
    return F.gelu(y, approximate="tanh") if gelu else y


def int8_linear(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], w_q: torch.Tensor, w_s: torch.Tensor,
                bias: Optional[torch.Tensor] = None, *, gelu: bool = False, path: Optional[str] = None,
                out_int8: bool = False, inv_smooth: Optional[torch.Tensor] = None):
    """y = x @ W^T + b [GELU tanh] with W = w_q * w_s[:, None]; x: bf16 [..., K] or a pre-quantised (x_q [M, K],
    x_s) pair (x_s [M]: per row; x_s [M, K / 128]: group scales, the "group" path). Returns bf16 [..., N], or
    (int8 [M, N], scales) with out_int8 (path "triton"/"group": [M, N / 128] group scales; "intmm": [M] per row)."""
    pre = isinstance(x, tuple)
    K = w_q.shape[1]
    N = w_q.shape[0]
    lead = None if pre else x.shape[:-1]
    if not enabled(x[0] if pre else x):
        if pre:
            x2 = dequantize_rows(*x).to(torch.bfloat16)
        else:
            x2 = x.reshape(-1, K)
            if inv_smooth is not None:
                x2 = (x2.float() * inv_smooth.float()).to(torch.bfloat16)
        y = _fallback(x2, w_q, w_s, bias, gelu)
        if out_int8:
            return quantize_rows_torch(y)
        return y if pre else y.view(*lead, N)
    path = path or PATHS.get((N, K), DEFAULT_PATH)
    if pre:
        xq, xs = x
        group = (K // xs.shape[1]) if xs.dim() == 2 else 0
        if group and path != "group":
            path = "group"
        if not group and path == "group":
            path = "triton"
    else:
        group = GROUP if path == "group" else 0
        xq, xs = quantize_rows(x, group, inv_smooth)
    if path == "intmm":
        y = intmm_linear(xq, xs, w_q, w_s, bias, gelu=gelu, out_int8=out_int8)
    else:
        y = gemm(xq, xs, w_q, w_s, bias, gelu=gelu, out="int8" if out_int8 else "bf16", group=group)
    if out_int8 or pre:
        return y
    return y.view(*lead, N)


def int8_mlp(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], w1: tuple, w2: tuple, *,
             path: str = "rows", cfg1: Optional[tuple] = None) -> torch.Tensor:
    """fc2(GELU(fc1(x))): w1, w2 = (w_q, w_s, bias).
    "rows" (fastest measured: 15.5 ms at M = 21504 vs 41.4 bf16): fc1 Triton with GELU -> bf16, quantize_rows,
           fc2 Triton (per-row scales everywhere, like the other layers);
    "group": fc1's Triton epilogue quantises each 128-column tile per row (int8 + group scales, the [M, 8192]
           hidden layer never in bf16; fc2 runs the group GEMM): 15.7 ms, 237 MiB peak vs 672 MiB bf16;
    "intmm": torch._int_mm, fc1's epilogue quantises each full row for fc2's torch._int_mm: 16.6 ms."""
    if path == "rows":
        xq, xs = x if isinstance(x, tuple) else quantize_rows(x)
        h = gemm(xq, xs, *w1, gelu=True, group=GROUP if xs.dim() == 2 else 0)
        return gemm(*quantize_rows(h), *w2)
    if path == "intmm":
        if isinstance(x, tuple):
            xq, xs = x
        else:
            xq, xs = quantize_rows(x)
        hq, hs = intmm_linear(xq, xs, *w1, gelu=True, out_int8=True)
        return intmm_linear(hq, hs, *w2)
    if isinstance(x, tuple):
        xq, xs = x
    else:
        xq, xs = quantize_rows(x)
    M = xq.shape[0]
    if cfg1 is None:
        cfg1 = CFG_GROUP if xs.dim() == 2 else CFG_FC1_INT8
    if cfg1[1] != GROUP:                            # fc1's BLOCK_N is the group size of fc2's input
        cfg1 = (cfg1[0], GROUP) + tuple(cfg1[2:])
    hq, hs = gemm(xq, xs, *w1, gelu=True, out="int8", group=GROUP if xs.dim() == 2 else 0, cfg=cfg1)
    return gemm(hq, hs, *w2, group=GROUP, cfg=CFG_GROUP)


# =====================================================================================================================
# References, the first-use self-check, the switch
# =====================================================================================================================

def reference_int(xq: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """The exact integer product (fp64 holds every int8 x int8 sum below 2^53), as int64."""
    return (xq.double() @ w_q.double().t()).to(torch.int64)


def reference(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
              bias: Optional[torch.Tensor], gelu: bool = False) -> torch.Tensor:
    """fp64 y from the same quantised operands (per-row or group scales), unrounded."""
    K = xq.shape[1]
    if xs.dim() == 2:
        g = K // xs.shape[1]
        xd = xq.double().view(xq.shape[0], -1, g) * xs.double().view(xq.shape[0], -1, 1)
        y = xd.view(xq.shape) @ w_q.double().t()
        y = y * w_s.double()[None, :]
    else:
        y = (xq.double() @ w_q.double().t()) * xs.double()[:, None] * w_s.double()[None, :]
    if bias is not None:
        y = y + bias.double()[None, :]
    if gelu:
        y = 0.5 * y * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (y + 0.044715 * y ** 3)))
    return y


def reference_mag(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                  bias: Optional[torch.Tensor], rows: int = 128) -> torch.Tensor:
    """The magnitude the fp32 epilogue rounds against (fp64): per-row scales |acc * rs * cs| + |bias|; group scales
    sum over the K groups of |partial_g * s_g| * cs + |bias| (the fp32 accumulator adds those terms)."""
    M, K = xq.shape
    out = []
    for r0 in range(0, M, rows):
        q = xq[r0:r0 + rows].double()
        if xs.dim() == 2:
            G = xs.shape[1]
            part = torch.einsum("mgk,ngk->mng", q.view(q.shape[0], G, -1), w_q.double().view(w_q.shape[0], G, -1))
            m = (part.abs() * xs[r0:r0 + rows].double()[:, None, :]).sum(-1) * w_s.double()[None, :]
        else:
            m = (q @ w_q.double().t()).abs() * xs[r0:r0 + rows].double()[:, None] * w_s.double()[None, :]
        if bias is not None:
            m = m + bias.double().abs()[None, :]
        out.append(m)
    return torch.cat(out)


def within(got: torch.Tensor, want64: torch.Tensor, mag64: torch.Tensor, c: float = 4.0,
           gelu: bool = False) -> Tuple[float, float]:
    """(largest |got - want| / (one bf16 ulp of want + c * 2^-24 * mag), share of elements != bf16(want)): <= 1 means
    the result is the fp64 value of the same int8 operands up to the epilogue's fp32 rounding (c roundings of the
    magnitude it works on: the int32 -> fp32 conversion, the two scale products, the bias add; + one per group for
    the group path's fp32 accumulator) and the final bf16 rounding. Through GELU (slope <= 1.13) the bound x 1.2."""
    w = want64.float()
    ulp = torch.where(w == 0, torch.full_like(w, 2.0 ** -133),
                      torch.exp2(torch.floor(torch.log2(w.abs())) - 7)).double()
    tol = ulp + c * (1.2 if gelu else 1.0) * 2.0 ** -24 * mag64
    d = (got.double() - want64).abs() / tol
    neq = (got.float() != w.to(torch.bfloat16).float()).double().mean()
    return float(d.max()), float(neq)


def bf16_ulps(got: torch.Tensor, want64: torch.Tensor) -> Tuple[float, float]:
    """(largest |got - want| in bf16 ulps of want, share of elements != bf16(want)). Misleading where the epilogue's
    terms cancel (a tiny result): use within()."""
    w = want64.float()
    ulp = torch.where(w == 0, torch.full_like(w, 2.0 ** -133),
                      torch.exp2(torch.floor(torch.log2(w.abs())) - 7))
    d = (got.double() - want64).abs() / ulp.double()
    neq = (got.float() != w.to(torch.bfloat16).float()).double().mean()
    return float(d.max()), float(neq)


def self_check(device: torch.device) -> Tuple[bool, str]:
    """Every path once on small tables (M = 300: masked tails), against fp64 from the same int8 operands: the
    integer part exactly, the bf16 outputs within the epilogue's fp32 rounding (within() <= 1). Done once per
    process before the first int8 call."""
    try:
        g = torch.Generator(device=device).manual_seed(0)
        M, K, N = 300, 256, 384
        x = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
        w = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g) * 0.05
        b = torch.randn((N,), device=device, dtype=torch.bfloat16, generator=g)
        xq, xs = quantize_rows(x)
        rq, rs = quantize_rows_torch(x)
        if not (torch.equal(xq, rq) and torch.equal(xs, rs)):
            return False, "quantize_rows differs from the torch reference"
        wq, ws = quantize_weight(w)
        exact = reference_int(xq, wq)
        tri = gemm(xq, xs, wq, ws, None, out="int32", cfg=(64, 64, 64, 8, 3, 4))
        if not torch.equal(tri.to(torch.int64), exact):
            return False, "the Triton int8 GEMM's integer product is wrong"
        if not torch.equal(torch._int_mm(xq, wq.t()).to(torch.int64), exact):
            return False, "torch._int_mm's integer product is wrong"
        mag = reference_mag(xq, xs, wq, ws, b)
        for gelu in (False, True):
            want = reference(xq, xs, wq, ws, b, gelu)
            for name, got in (("triton", gemm(xq, xs, wq, ws, b, gelu=gelu, cfg=(64, 64, 64, 8, 3, 4))),
                              ("intmm", intmm_linear(xq, xs, wq, ws, b, gelu=gelu))):
                worst, _ = within(got, want, mag, gelu=gelu)
                if worst > 1.0:
                    return False, f"{name} (gelu={gelu}) is {worst:.2f}x the fp32-rounding bound off"
        gq, gs = quantize_rows(x, GROUP)
        rq, rs = quantize_rows_torch(x, GROUP)
        if not (torch.equal(gq, rq) and torch.equal(gs, rs)):
            return False, "group quantize_rows differs from the torch reference"
        got = gemm(gq, gs, wq, ws, b, group=GROUP, cfg=(64, 64, 128, 8, 3, 4))
        worst, _ = within(got, reference(gq, gs, wq, ws, b), reference_mag(gq, gs, wq, ws, b), c=4 + K // GROUP)
        if worst > 1.0:
            return False, f"group GEMM {worst:.2f}x the fp32-rounding bound off"
        torch.cuda.synchronize(device)
    except Exception as exc:        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    return True, ""


def ready(device: torch.device) -> Tuple[bool, str]:
    """Whether the kernels run on this card (the capability gate, then the self-check once per process): (yes,
    the reason when not). Cached."""
    if _state["ok"] is None:
        ok_mm, ok_triton, why = capability(device)
        if not (ok_mm and ok_triton):
            _state["ok"], _state["why"] = False, why
        else:
            _state["ok"], _state["why"] = self_check(device)
    return bool(_state["ok"]), str(_state["why"])


def enabled(x: torch.Tensor) -> bool:
    """Whether the int8 path serves x (CUDA, no autograd, capable card, self-check passed); else the fallback.
    (The switch PIXAL3D_INT8 is read by modules/int8.py, which converts layers only when it says so; a converted
    layer always runs the kernels where they can run.)"""
    if not x.is_cuda or torch.is_grad_enabled():
        return False
    ok, why = ready(x.device)
    if not ok and not _state["warned"]:
        _state["warned"] = True
        print(f"[INT8] the int8 linear path is off ({why}); F.linear serves", flush=True)
    return ok
