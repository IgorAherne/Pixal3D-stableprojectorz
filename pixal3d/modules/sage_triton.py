"""SageAttention's preprocessing in one Triton pass, and its Triton attention kernel without length specialisation,
for modules/sage.py.

The SageAttention wheel (thu-ml/SageAttention 2.2.0, Windows build by woct0rdho; Apache-2.0) prepares q, k and v in
three to five passes over the tables before its attention kernel runs: k's mean over the tokens (torch), k minus that
mean (the Triton path: a bf16 table), the int8 quantisation of q and of k (one kernel each), and v cast to fp16. The
kernels here do the same arithmetic in ONE pass after torch's k.mean (read q, k, v once, write the int8 codes, their
scales and v in fp16) and give the same bits as the wheel's own pieces (torch.equal on the flows' real tensors and on
random tables: code/devtools/pixal3d-lowmem/round-j/sagefast in Spellcast3D):

  ship_prep(q, k, km, v)    -> q8, q_scale, k8, k_scale, v16 for the wheel's sm80 CUDA kernel
                               (qk_int8_sv_f16_accum_f16_attn_inst_buf, per_warp, fp16+fp32): q per 16-token warp,
                               k minus its mean per 64-token block, v to fp16. The arithmetic is the wheel's compiled
                               code's, instruction for instruction: the scale stored is amax * fp32(1/127), the codes
                               round x * (rcp.approx(amax) * 127) to the nearest even (cvt.rni), amax starts from 1e-7.
  ship(q, k, v)             sageattn_qk_int8_pv_fp16_cuda(per_warp, fp16+fp32) with ship_prep and the wheel's own
                               sm80 kernel: the same answer to the last bit.
  triton_prep(q, k, km, v)  -> q8, q_scale, k8, k_scale, v16 as the wheel's Triton path makes them (per_block_int8:
                               q per 128-token block scaled by sm_scale * log2(e), k minus its mean - rounded to bf16
                               as torch's subtraction rounds it - per 64-token block; v.to(torch.float16)).
  triton(q, k, v)           sageattn_qk_int8_pv_fp16_triton(quantization_backend "triton") with triton_prep and the
                               wheel's own Triton kernel _attn_fwd (BLOCK_M 128, BLOCK_N 64, 8 warps, 4 stages at head
                               size 128), compiled once more with Triton's specialisation of its two length arguments
                               turned off: the wheel's kernel is specialised for lengths divisible by 16 and that
                               variant is 7-10 % SLOWER on an RTX 3060 (measured, round J), so the generic code serves
                               every N (and no N compiles a second variant). The same answer to the last bit.

Measured and left out (round J): q quantised inside the attention kernel (each program owns one block of queries) -
no extra pass, but the kernel runs 4-7 % slower; v cast to fp16 inside the kernel - 11-12 % slower (every block of
queries converts all of v again). k's mean is torch's own `k.mean(dim=1, keepdim=True)` everywhere, as the wheel
computes it (the order of that sum decides its bits). Triton compiles each kernel on its first call (the fork's
self-check), once per process and card (then from the disk cache, TRITON_CACHE_DIR).
"""

from __future__ import annotations

from typing import Optional

import torch

LOG2E = 1.44269504          # the wheel's constant (core.py / quant.py)

_kernels: Optional[dict] = None


def _build() -> dict:
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def round_bf16(x):
        # x (fp32) rounded to the nearest bf16, ties to even, widened back (modules/fused.py: integer arithmetic on
        # the bits, so that the compiler cannot narrow or contract the rounding away); what torch's bf16 ops do
        bits = x.to(tl.uint32, bitcast=True)
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
        return tl.where(x != x, x, bits.to(tl.float32, bitcast=True))

    @triton.jit
    def rcp_approx(x):
        # MUFU.RCP: what the wheel's nvcc --use_fast_math build makes of 127.0f / amax (rcp.approx.ftz, then * 127)
        return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32,
                                         is_pure=True, pack=1)

    # --- the CUDA path: q per 16-token warp, k - mean per 64-token block, v -> fp16, one pass ----------------------
    @triton.jit(do_not_specialize=["L", "n_kblk"])
    def ship_prep_kernel(Q, K, KM, V, Q8, QS, K8, KS, V16, L, n_kblk,
                         s_qz, s_qn, s_qh, s_kz, s_kn, s_kh, s_vz, s_vn, s_vh, s_mz, s_mh,
                         s_oz, s_on, s_oh, s_v16z, s_v16n, s_v16h, s_qsz, s_qsh, s_ksz, s_ksh,
                         C: tl.constexpr):
        blk = tl.program_id(0)
        h = tl.program_id(1).to(tl.int64)
        z = tl.program_id(2).to(tl.int64)
        rows = blk * 64 + tl.arange(0, 64)
        cols = tl.arange(0, C)
        inside = rows < L
        mask = inside[:, None]
        # q: four warps of 16 tokens, one scale each (QuantInt8Kernel<128, 16, 1, false, false>)
        x = tl.load(Q + z * s_qz + h * s_qh + rows[:, None] * s_qn + cols[None, :], mask=mask, other=0.0)
        x = x.to(tl.float32)
        row_max = tl.max(tl.abs(x), axis=1)
        amax = tl.maximum(tl.max(tl.reshape(row_max, (4, 16)), axis=1), 1e-7)
        tl.store(QS + z * s_qsz + h * s_qsh + blk * 4 + tl.arange(0, 4), amax * 0.0078740157186985015869140625)
        inv = rcp_approx(amax) * 127.0
        inv_rows = tl.reshape(tl.broadcast_to(inv[:, None], (4, 16)), (64,))
        q8 = libdevice.rint(x * inv_rows[:, None]).to(tl.int8)
        tl.store(Q8 + z * s_oz + h * s_oh + rows[:, None] * s_on + cols[None, :], q8, mask=mask)
        # k minus its mean, one scale per 64 tokens (QuantInt8Kernel<128, 64, 1, false, true>)
        km = tl.load(KM + z * s_mz + h * s_mh + cols).to(tl.float32)
        x = tl.load(K + z * s_kz + h * s_kh + rows[:, None] * s_kn + cols[None, :], mask=mask, other=0.0)
        x = tl.where(mask, x.to(tl.float32) - km[None, :], 0.0)
        amax_k = tl.maximum(tl.max(tl.max(tl.abs(x), axis=1), axis=0), 1e-7)
        amax_k1 = tl.full((1,), 0.0, tl.float32) + amax_k
        tl.store(KS + z * s_ksz + h * s_ksh + blk + tl.arange(0, 1), amax_k1 * 0.0078740157186985015869140625,
                 mask=blk < n_kblk)
        inv_k = rcp_approx(amax_k1) * 127.0
        k8 = libdevice.rint(x * inv_k[:, None]).to(tl.int8)
        tl.store(K8 + z * s_oz + h * s_oh + rows[:, None] * s_on + cols[None, :], k8, mask=mask)
        # v to fp16 (torch's v.to(torch.float16): round to nearest even)
        v = tl.load(V + z * s_vz + h * s_vh + rows[:, None] * s_vn + cols[None, :], mask=mask, other=0.0)
        tl.store(V16 + z * s_v16z + h * s_v16h + rows[:, None] * s_v16n + cols[None, :], v.to(tl.float16), mask=mask)

    # --- the Triton path: q per 128-token block, k - mean (bf16) per 64-token block, v -> fp16, one pass -----------
    @triton.jit
    def quant_block(x, sm_scale):
        # quant_per_block_int8_kernel's arithmetic on one block (x fp32, rows past the end already 0)
        x *= sm_scale
        scale = tl.max(tl.abs(x)) / 127.
        x_int8 = x / scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        return x_int8.to(tl.int8), scale

    @triton.jit(do_not_specialize=["L", "n_kblk"])
    def triton_prep_kernel(Q, K, KM, V, Q8, QS, K8, KS, V16, L, n_kblk, q_scale_mul,
                           s_qz, s_qn, s_qh, s_kz, s_kn, s_kh, s_vz, s_vn, s_vh, s_mz, s_mh,
                           s_oz, s_on, s_oh, s_v16z, s_v16n, s_v16h, s_qsz, s_qsh, s_ksz, s_ksh,
                           C: tl.constexpr):
        blk = tl.program_id(0)
        h = tl.program_id(1).to(tl.int64)
        z = tl.program_id(2).to(tl.int64)
        cols = tl.arange(0, C)
        # q: one block of 128 tokens (per_block_int8, BLKQ 128, scaled by sm_scale * log2(e))
        rows = blk * 128 + tl.arange(0, 128)
        mask = rows[:, None] < L
        x = tl.load(Q + z * s_qz + h * s_qh + rows[:, None] * s_qn + cols[None, :], mask=mask)
        q8, qs = quant_block(x.to(tl.float32), q_scale_mul)
        tl.store(Q8 + z * s_oz + h * s_oh + rows[:, None] * s_on + cols[None, :], q8, mask=mask)
        tl.store(QS + z * s_qsz + h * s_qsh + blk, qs)
        # v: the same 128 tokens to fp16
        v = tl.load(V + z * s_vz + h * s_vh + rows[:, None] * s_vn + cols[None, :], mask=mask)
        tl.store(V16 + z * s_v16z + h * s_v16h + rows[:, None] * s_v16n + cols[None, :], v.to(tl.float16), mask=mask)
        # k: the two blocks of 64 tokens in them (BLKK 64), k - km rounded to bf16 as torch's subtraction leaves it
        km = tl.load(KM + z * s_mz + h * s_mh + cols).to(tl.float32)
        for j in tl.static_range(2):
            kb = blk * 2 + j
            krows = kb * 64 + tl.arange(0, 64)
            kmask = krows[:, None] < L
            xk = tl.load(K + z * s_kz + h * s_kh + krows[:, None] * s_kn + cols[None, :], mask=kmask, other=0.0)
            xk = tl.where(kmask, round_bf16(xk.to(tl.float32) - km[None, :]), 0.0)
            k8, ks = quant_block(xk, 1.0)
            tl.store(K8 + z * s_oz + h * s_oh + krows[:, None] * s_on + cols[None, :], k8, mask=kmask)
            tl.store(KS + z * s_ksz + h * s_ksh + kb, ks, mask=kb < n_kblk)

    from sageattention.triton import attn_qk_int8_per_block as wheel_attn

    # the wheel's own attention kernel (its source, its helper), compiled without the length specialisation
    attn = triton.jit(do_not_specialize=["qo_len", "kv_len"])(wheel_attn._attn_fwd.fn)
    _kernels = {"ship_prep": ship_prep_kernel, "triton_prep": triton_prep_kernel, "attn": attn,
                "no_lse": torch.empty([0], dtype=torch.float32, device="cpu"), "cdiv": triton.cdiv}
    return _kernels


def _prep_launch(kernel, grid, q, k, km, v, q8, q_scale, k8, k_scale, v16, n, n_kblk, *extra):
    kernel[grid](
        q, k, km, v, q8, q_scale, k8, k_scale, v16, n, n_kblk, *extra,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), km.stride(0), km.stride(1),
        q8.stride(0), q8.stride(1), q8.stride(2), v16.stride(0), v16.stride(1), v16.stride(2),
        q_scale.stride(0), q_scale.stride(1), k_scale.stride(0), k_scale.stride(1),
        C=q.shape[-1], num_warps=4)


# =====================================================================================================================
# The CUDA path (sm_8x): the wheel's per_warp quantisation and v cast in one pass, then its sm80 kernel
# =====================================================================================================================

def ship_prep(q: torch.Tensor, k: torch.Tensor, km: torch.Tensor, v: torch.Tensor):
    """q, k, v [B, N, H, 128] (NHD, last dimension contiguous; v may be a strided view), km = k.mean(dim=1,
    keepdim=True): (q8, q_scale, k8, k_scale, v16) exactly as sageattention.quant.per_warp_int8(q, k, km, "NHD",
    BLKQ=128, WARPQ=16, BLKK=64) and v.to(torch.float16) make them."""
    kn = _build()
    b, n, h, d = q.shape
    q8 = torch.empty((b, n, h, d), dtype=torch.int8, device=q.device)
    k8 = torch.empty((b, n, h, d), dtype=torch.int8, device=q.device)
    v16 = torch.empty((b, n, h, d), dtype=torch.float16, device=q.device)
    n128 = (n + 127) // 128
    q_scale = torch.empty((b, h, n128 * 8), dtype=torch.float32, device=q.device)
    n_kblk = (n + 63) // 64
    k_scale = torch.empty((b, h, n_kblk), dtype=torch.float32, device=q.device)
    _prep_launch(kn["ship_prep"], (n128 * 2, h, b), q, k, km.reshape(b, h, d), v, q8, q_scale, k8, k_scale, v16,
                 n, n_kblk)
    return q8, q_scale, k8, k_scale, v16


def ship(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """sageattn_qk_int8_pv_fp16_cuda(q, k, v, "NHD", per_warp, fp16+fp32), not causal, head size 128, with the
    preprocessing in one pass: the same answer to the last bit."""
    import sageattention.core  # noqa: F401  (registers torch.ops.sageattention_qattn_sm80)

    km = k.mean(dim=1, keepdim=True)
    q8, q_scale, k8, k_scale, v16 = ship_prep(q, k, km, v)
    o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    torch.ops.sageattention_qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(
        q8, k8, v16, o, q_scale, k_scale, 0, 0, 2, q.size(-1) ** -0.5, 0)
    return o


# =====================================================================================================================
# The Triton path: the wheel's Triton kernel, unspecialised, its preprocessing in one pass
# =====================================================================================================================

def triton_prep(q: torch.Tensor, k: torch.Tensor, km: torch.Tensor, v: torch.Tensor):
    """q, k, v [B, N, H, 128] NHD (v may be a strided view), km = k.mean(dim=1, keepdim=True): (q8, q_scale, k8,
    k_scale, v16) exactly as the wheel's per_block_int8(q, k, km, BLKQ=128, BLKK=64, sm_scale=128 ** -0.5) and
    v.to(torch.float16) make them."""
    kn = _build()
    b, n, h, d = q.shape
    q8 = torch.empty((b, n, h, d), dtype=torch.int8, device=q.device)
    k8 = torch.empty((b, n, h, d), dtype=torch.int8, device=q.device)
    v16 = torch.empty((b, n, h, d), dtype=torch.float16, device=q.device)
    n128 = (n + 127) // 128
    q_scale = torch.empty((b, h, n128), dtype=torch.float32, device=q.device)
    n_kblk = (n + 63) // 64
    k_scale = torch.empty((b, h, n_kblk), dtype=torch.float32, device=q.device)
    sm_scale = 1.0 / (d ** 0.5)         # sageattn_qk_int8_pv_fp16_triton's default, then * log2(e) as quant.py does
    _prep_launch(kn["triton_prep"], (n128, h, b), q, k, km.reshape(b, h, d), v, q8, q_scale, k8, k_scale, v16,
                 n, n_kblk, sm_scale * LOG2E)
    return q8, q_scale, k8, k_scale, v16


def triton_attn(q8: torch.Tensor, k8: torch.Tensor, v16: torch.Tensor, q_scale: torch.Tensor, k_scale: torch.Tensor,
                out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The wheel's Triton attention kernel (NHD, no mask, no lse; its launch: BLOCK_M 128, BLOCK_N 64, 8 warps,
    4 stages at head size 128), compiled without the length specialisation."""
    kn = _build()
    b, qo_len, h, d = q8.shape
    kv_len, h_kv = k8.shape[1], k8.shape[2]
    o = torch.empty(q8.shape, dtype=out_dtype, device=q8.device)
    kn["attn"][(kn["cdiv"](qo_len, 128), h, b)](
        q8, k8, v16, q_scale, k_scale, o, None, kn["no_lse"],
        q8.stride(0), q8.stride(2), q8.stride(1), k8.stride(0), k8.stride(2), k8.stride(1),
        v16.stride(0), v16.stride(2), v16.stride(1), o.stride(0), o.stride(2), o.stride(1),
        0, 0, 0, 0, qo_len, kv_len, h, h // h_kv,
        BLOCK_M=128, BLOCK_N=64, HEAD_DIM=d, STAGE=1, RETURN_LSE=False,
        num_warps=4 if d == 64 else 8, num_stages=3 if d == 64 else 4)
    return o


def triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """sageattn_qk_int8_pv_fp16_triton(q, k, v, "NHD", quantization_backend="triton"), not causal, head size 128:
    torch's k.mean, triton_prep, the unspecialised kernel - the same answer to the last bit."""
    km = k.mean(dim=1, keepdim=True)
    q8, q_scale, k8, k_scale, v16 = triton_prep(q, k, km, v)
    return triton_attn(q8, k8, v16, q_scale, k_scale, out_dtype=q.dtype)
