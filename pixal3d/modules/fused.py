"""Fused elementwise kernels (Triton) for the flow transformers' blocks, behind a switch that is ON by default.

Every block of the four flows (sparse: lr, hr, tex - modules/sparse/transformer/modulated.py; dense: ss -
modules/transformer/modulated.py) runs a dozen elementwise PyTorch ops per sub-layer, each a full read and write of
the [tokens, 1536] features (or of q / k in fp32): LayerNorm32's two casts, the adaLN modulation, the gate, the
residual add, the q / k RMS norm and the rotary embedding. On a compute-bound flow that traffic is ~18 % of a pass.
The kernels here do each group in one read and one write:

    norm_modulate   LayerNorm32(x) [* weight + bias] [* (1 + scale) + shift]                    -> x.dtype
    residual        x + gate * h,  x + h,  x + (h2 + h)                                         -> x.dtype
    qk_norm_rope    F.normalize(q.float()) * gamma * sqrt(D), then the rotary embedding           -> q.dtype

The switch is the environment variable PIXAL3D_FUSED, read once when this module is first imported (set_mode()
changes it in a running process, for checks that compare both paths on one loaded network):

    unset, "", "1", "on", "rounded"
                            fused (the default); inside the kernel every value is rounded to bf16 where the eager
                            op chain
                            rounds it (after LayerNorm32, after 1 + scale, after each product and sum, after the RMS
                            norm), so only the two reductions (LayerNorm's mean and variance, the norm of q / k) are
                            summed in another order. Measured on the hr flow's shapes against the eager ops: the
                            residual kernels bit-identical, 4e-6 of the modulated LayerNorm's outputs, 2e-5 of the
                            affine one's and 3e-6 of q / k one bf16 ulp off.
    "fp32"                  fused; fp32 inside, one rounding at the store: more exact than the eager chain, and
                            therefore unlike it in 4-36 % of the elements (one ulp).
    "0", "off"              off: the eager PyTorch code, unchanged. The golden checks (check_sampler) run this way.
    anything else           off, with a warning.

What the kernels change is the rounding of the elementwise chain's reductions, not the arithmetic: the samplers
amplify any such change - as they amplify a different GPU or cuBLAS - into a slightly different sample of the same
quality (measured on the five test props at 1536: the same designs, poses and textures, fine detail differing;
Chamfer distance 0.2-0.4 % of the bounding-box diagonal). A pass of the detail flow is 11-14 % faster, of the
shape flow 21 %, a batch 8-10 %. The memory peak of a pass is unchanged.

The fused path is taken only where it applies: CUDA tensors in bf16, no autograd, no autocast, the shapes the flows
use; anything else runs the eager code. Before the first fused call of a process the three kernels run once on small
tables of the flows' widths, against the eager ops (`_check`): a card or a Triton that cannot compile or run them,
or runs them wrong, turns the switch off with a warning, and the flows run their eager code. Triton compiles each
kernel on its first call (about 0.1 s each, 1.5 s for all of them) and keeps it in its cache for the next
processes (the launcher's TRITON_CACHE_DIR).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

_MODES = {"0": "off", "off": "off", "false": "off", "no": "off",
          "": "rounded", "1": "rounded", "on": "rounded", "true": "rounded", "yes": "rounded", "rounded": "rounded",
          "fp32": "fp32"}


def _mode_from_env() -> str:
    text = os.environ.get("PIXAL3D_FUSED", "").strip().lower()
    if text not in _MODES:
        print(f"[FUSED] PIXAL3D_FUSED={text!r} is none of off, rounded (1), fp32: the fused kernels stay off",
              flush=True)
        return "off"
    return _MODES[text]


MODE = _mode_from_env()


def set_mode(mode: str) -> None:
    """Switch the fused path for the running process: "off", "rounded" (or "1"), "fp32"."""
    global MODE
    key = str(mode).strip().lower()
    if key not in _MODES:
        raise ValueError(f"unknown fused mode {mode!r}: expected off, rounded (1), fp32")
    MODE = _MODES[key]


_checked = False        # the kernels ran once on this process's card and agreed with the eager ops (_check)


def active(feats: torch.Tensor) -> bool:
    """Whether the blocks take the fused path for these features (bf16 only: the flows' dtype, and the one the
    rounded mode reproduces). The first time, the kernels are tried on the card (`_check`)."""
    if (MODE == "off" or not feats.is_cuda or feats.dtype != torch.bfloat16 or torch.is_grad_enabled()
            or torch.is_autocast_enabled()):
        return False
    return _checked or _check(feats.device)


def _check(device: torch.device) -> bool:
    """The three kernels once, on small tables of the flows' widths, against the eager ops (a bf16 ulp or two of
    leeway: the reductions are summed in another order). Compiling or running them may fail on a card or a Triton
    the kernels were not made for, and a kernel that is miscompiled gives other values: either turns the switch
    off, with a warning, so that the flows run their eager code instead of failing the prop."""
    global _checked, MODE
    try:
        from .norm import LayerNorm32

        if _build() is None:
            raise RuntimeError("Triton is not installed")
        g = torch.Generator(device=device).manual_seed(0)
        C, H, D, rows = 1536, 12, 128, 48
        x = torch.randn((rows, C), device=device, dtype=torch.bfloat16, generator=g)
        h = torch.randn((rows, C), device=device, dtype=torch.bfloat16, generator=g)
        scale = torch.randn((1, C), device=device, dtype=torch.bfloat16, generator=g)
        shift = torch.randn((1, C), device=device, dtype=torch.bfloat16, generator=g)
        norm = LayerNorm32(C).to(device)
        with torch.no_grad():
            norm.weight.normal_(generator=g)
            norm.bias.normal_(generator=g)
        eager_ln = F.layer_norm(x.float(), (C,), eps=norm.eps).to(x.dtype)
        close(norm_modulate(x, norm.eps, scale=scale, shift=shift, rows_per_batch=rows),
              eager_ln * (1 + scale) + shift, "modulated LayerNorm")
        close(norm_modulate(x, norm.eps, weight=norm.weight, bias=norm.bias, rows_per_batch=rows), norm(x),
              "affine LayerNorm")
        close(residual(x, h, gate=scale, rows_per_batch=rows), x + h * scale, "gated residual")
        close(residual(x, h, second=x), x + (x + h), "summed residual")
        q = torch.randn((rows, H, D), device=device, dtype=torch.bfloat16, generator=g)
        gamma = torch.randn((H, D), device=device, generator=g)
        angles = torch.rand((rows, D // 2), device=device, generator=g) * 6.2832
        phases = torch.polar(torch.ones_like(angles), angles)
        normed = (F.normalize(q.float(), dim=-1) * gamma * D ** 0.5).to(q.dtype)
        close(qk_norm_rope(q, gamma, D ** 0.5, None), normed, "q/k RMS norm")
        rotated = torch.view_as_complex(normed.float().reshape(rows, H, D // 2, 2)) * phases.unsqueeze(-2)
        close(qk_norm_rope(q, gamma, D ** 0.5, phases), torch.view_as_real(rotated).reshape(rows, H, D).to(q.dtype),
              "q/k RMS norm + RoPE")
        torch.cuda.synchronize(device)
    except Exception as exc:        # noqa: BLE001 - whatever Triton or the card raises: the eager path serves
        MODE = "off"
        print(f"[FUSED] the fused kernels do not run on this card ({type(exc).__name__}: {str(exc)[:300]}); "
              f"the flows run their eager code", flush=True)
        return False
    _checked = True
    print(f"[FUSED] the flows' blocks run fused elementwise kernels (PIXAL3D_FUSED: {MODE})", flush=True)
    return True


def close(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    """`got` within two bf16 ulps of `want` (the eager result) everywhere, else a RuntimeError naming `what`."""
    diff = (got.float() - want.float()).abs()
    allowed = 2 * 2.0 ** -8 * want.float().abs() + 2.0 ** -6
    if got.shape != want.shape or not bool((diff <= allowed).all()):
        raise RuntimeError(f"{what}: the fused kernel's values differ from the eager ops' "
                           f"(largest difference {float(diff.max()):.4g})")


# =====================================================================================================================
# The kernels: defined when this module is imported with the switch on (else on the first fused call, after a
# set_mode): the eager path never imports Triton from here. Triton reads a kernel's source from the file when the
# kernel is defined, so defining them at import keeps an edit of this file during a run from reaching a kernel.
# =====================================================================================================================

_kernels = None


def _build():
    """The kernels (a dict), defined once; None, with the switch turned off, where Triton cannot be imported."""
    global _kernels, MODE
    if _kernels is not None:
        return _kernels
    try:
        import triton
        import triton.language as tl
    except Exception as exc:        # noqa: BLE001 - no Triton, or one that does not load: the eager path serves
        MODE = "off"
        print(f"[FUSED] Triton is not available ({type(exc).__name__}: {str(exc)[:200]}); the flows run their eager "
              f"code", flush=True)
        return None

    @triton.jit
    def round_bf16(x):
        # x (fp32) rounded to the nearest bf16, ties to even, and widened back: what each eager bf16 op does to its
        # fp32 result. Integer arithmetic on the bits, not x.to(bf16).to(fp32): the compiler narrows that round
        # trip after a product of two bf16 values to a bf16 multiply and then contracts it with the next add into
        # one FMA, which skips the rounding. NaN is passed through.
        bits = x.to(tl.uint32, bitcast=True)
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
        return tl.where(x != x, x, bits.to(tl.float32, bitcast=True))

    @triton.jit(do_not_specialize=["n_rows", "rows_per_batch"])
    def norm_modulate_kernel(X, Y, W, B, SCALE, SHIFT, BIDX, n_rows, rows_per_batch, C, stride_x, stride_y,
                             stride_mod, eps, AFFINE: tl.constexpr, MODULATE: tl.constexpr,
                             HAS_BIDX: tl.constexpr, ROUNDED: tl.constexpr, BLOCK_C: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        dt = Y.dtype.element_ty
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
                # LayerNorm32's cast back, then the three eager ops: 1 + scale, h * (1 + scale), + shift
                y = round_bf16(y)
                s1 = round_bf16(scale + 1.0)
                y = round_bf16(y * s1)
                y = y + shift
            else:
                y = y * (scale + 1.0) + shift
        tl.store(Y + row * stride_y + cols, y.to(dt), mask=mask)

    @triton.jit(do_not_specialize=["n", "rows_per_batch"])
    def residual_kernel(X, H, H2, G, BIDX, Y, n, rows_per_batch, C, stride_g, GATE: tl.constexpr,
                        SECOND: tl.constexpr, HAS_BIDX: tl.constexpr, ROUNDED: tl.constexpr,
                        BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        dt = Y.dtype.element_ty
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(H + offs, mask=mask, other=0.0).to(tl.float32)
        if GATE:
            row = offs // C
            col = offs - row * C
            if HAS_BIDX:
                bi = tl.load(BIDX + row, mask=mask, other=0).to(tl.int64)
            else:
                bi = row // rows_per_batch
            g = tl.load(G + bi * stride_g + col, mask=mask, other=0.0).to(tl.float32)
            t = t * g
            if ROUNDED:
                t = round_bf16(t)
        if SECOND:
            t = tl.load(H2 + offs, mask=mask, other=0.0).to(tl.float32) + t
            if ROUNDED:
                t = round_bf16(t)
        tl.store(Y + offs, (x + t).to(dt), mask=mask)

    @triton.jit(do_not_specialize=["n_rows", "phase_period"])
    def qk_kernel(X, Y, GAMMA, PH, n_rows, H, phase_period, stride_xt, scale, NORM: tl.constexpr,
                  ROPE: tl.constexpr, ROUNDED: tl.constexpr, D: tl.constexpr, BLOCK_R: tl.constexpr):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < n_rows
        tok = rows // H
        head = rows - tok * H
        dt = Y.dtype.element_ty
        dims = tl.arange(0, D)
        x = tl.load(X + tok[:, None] * stride_xt + head[:, None] * D + dims[None, :], mask=rmask[:, None],
                    other=0.0).to(tl.float32)
        if NORM:
            # F.normalize(x.float(), dim=-1) * gamma * scale: x / max(||x||, 1e-12), with the IEEE division and
            # square root PyTorch's kernels use
            nrm = tl.sqrt_rn(tl.sum(x * x, axis=1))
            den = tl.maximum(nrm, 1e-12)
            x = tl.div_rn(x, tl.broadcast_to(den[:, None], (BLOCK_R, D)))
            g = tl.load(GAMMA + head[:, None] * D + dims[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
            x = (x * g) * scale
            if ROUNDED:
                x = round_bf16(x)
        if ROPE:
            # (a + bi) * (c + di) with the phases of the token (complex64 as float pairs)
            pidx = tok % phase_period
            p = tl.load(PH + pidx[:, None] * D + dims[None, :], mask=rmask[:, None], other=0.0)
            a, b = tl.split(tl.reshape(x, (BLOCK_R, D // 2, 2)))
            c, d = tl.split(tl.reshape(p, (BLOCK_R, D // 2, 2)))
            re = a * c - b * d
            im = a * d + b * c
            x = tl.reshape(tl.join(re, im), (BLOCK_R, D))
        tl.store(Y + rows[:, None] * D + dims[None, :], x.to(dt), mask=rmask[:, None])

    _kernels = {"norm_modulate": norm_modulate_kernel, "residual": residual_kernel, "qk": qk_kernel,
                "next_pow2": triton.next_power_of_2, "cdiv": triton.cdiv}
    return _kernels


# =====================================================================================================================
# The calls the blocks make
# =====================================================================================================================

def _mod_rows(mod: Optional[torch.Tensor], batch: int, C: int) -> tuple[Optional[torch.Tensor], int]:
    """A [B, C] (or [1, C]) modulation vector and its row stride (0 when one row serves every batch item)."""
    if mod is None:
        return None, 0
    if mod.dim() != 2:
        mod = mod.reshape(mod.shape[0], -1)
    if mod.shape[-1] != C or mod.stride(-1) != 1 or mod.shape[0] not in (1, batch):
        raise ValueError(f"modulation of shape {tuple(mod.shape)} for {batch} items of {C} channels")
    return mod, (0 if mod.shape[0] == 1 else mod.stride(0))


def norm_modulate(x: torch.Tensor, eps: float, *, weight: Optional[torch.Tensor] = None,
                  bias: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None,
                  shift: Optional[torch.Tensor] = None, batch: int = 1, rows_per_batch: Optional[int] = None,
                  batch_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """LayerNorm32(x) (affine when weight / bias are given), then * (1 + scale) + shift when they are given. Row r
    of x ([..., C], last dimension contiguous) belongs to batch item r // rows_per_batch, or batch_index[r]."""
    k = _build()
    C = x.shape[-1]
    flat = x.reshape(-1, C)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    n_rows = flat.shape[0]
    y = torch.empty((n_rows, C), dtype=x.dtype, device=x.device)
    scale, stride_mod = _mod_rows(scale, batch, C)
    shift, stride_shift = _mod_rows(shift, batch, C)
    if scale is not None and stride_mod != stride_shift:
        raise ValueError("scale and shift must have the same row stride")
    if n_rows:
        k["norm_modulate"][(n_rows,)](
            flat, y, weight if weight is not None else flat, bias if bias is not None else flat,
            scale if scale is not None else flat, shift if shift is not None else flat,
            batch_index if batch_index is not None else flat, n_rows, rows_per_batch or n_rows, C,
            flat.stride(0), y.stride(0), stride_mod, float(eps), AFFINE=weight is not None,
            MODULATE=scale is not None, HAS_BIDX=batch_index is not None, ROUNDED=MODE != "fp32",
            BLOCK_C=k["next_pow2"](C), num_warps=8 if C > 1024 else 4)
    return y.view(x.shape)


def residual(x: torch.Tensor, h: torch.Tensor, *, gate: Optional[torch.Tensor] = None,
             second: Optional[torch.Tensor] = None, batch: int = 1, rows_per_batch: Optional[int] = None,
             batch_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x + gate * h, or x + (second + h) (the projection attention's sum), or x + h: one rounding per eager op
    ("rounded") or one at the end ("fp32"). Every operand of x's shape."""
    k = _build()
    if h.shape != x.shape or (second is not None and second.shape != x.shape):
        raise ValueError("residual: operands of one shape only")
    x, h = x.contiguous(), h.contiguous()               # no copy for the blocks' tables, which are contiguous
    second = second.contiguous() if second is not None else None
    C = x.shape[-1]
    n = x.numel()
    y = torch.empty_like(x)
    gate, stride_g = _mod_rows(gate, batch, C)
    BLOCK = 2048
    if n:
        k["residual"][(k["cdiv"](n, BLOCK),)](
            x, h, second if second is not None else x, gate if gate is not None else x,
            batch_index if batch_index is not None else x, y, n, rows_per_batch or n // C, C, stride_g,
            GATE=gate is not None, SECOND=second is not None, HAS_BIDX=batch_index is not None,
            ROUNDED=MODE != "fp32", BLOCK=BLOCK, num_warps=8)
    return y


def qk_norm_rope(x: torch.Tensor, gamma: Optional[torch.Tensor], scale: float,
                 phases: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """The multi-head RMS norm of q or k ([..., H, D], heads packed, tokens evenly strided) when gamma is given, then
    the rotary embedding when phases (complex64 [P, D / 2]; token t takes row t % P) are given. Returns a contiguous
    tensor of x's shape, or None when x's layout does not suit the kernel (the caller then runs the eager code)."""
    H, D = x.shape[-2], x.shape[-1]
    if x.stride(-1) != 1 or x.stride(-2) != D or D & (D - 1) or D < 2:
        return None
    try:
        tokens = x.view(-1, H, D)
    except RuntimeError:
        return None
    if gamma is not None and (tuple(gamma.shape) != (H, D) or not gamma.is_contiguous()):
        return None
    ph = None
    if phases is not None:
        if phases.dtype != torch.complex64 or phases.shape[-1] != D // 2:
            return None
        ph = torch.view_as_real(phases.reshape(-1, D // 2))
        if not ph.is_contiguous():
            ph = ph.contiguous()
    k = _build()
    T = tokens.shape[0]
    y = torch.empty((T, H, D), dtype=x.dtype, device=x.device)
    BLOCK_R = 16
    if T:
        k["qk"][(k["cdiv"](T * H, BLOCK_R),)](
            tokens, y, gamma if gamma is not None else tokens, ph if ph is not None else tokens, T * H, H,
            ph.shape[0] if ph is not None else 1, tokens.stride(0), float(scale), NORM=gamma is not None,
            ROPE=ph is not None, ROUNDED=MODE != "fp32", D=D, BLOCK_R=BLOCK_R, num_warps=4)
    return y.view(x.shape)


if MODE != "off":
    _build()
