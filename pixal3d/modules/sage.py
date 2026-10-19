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
    "cuda", "triton", "wheel"    on, with the sm_8x kernel forced (developers): the CUDA kernel fed by the fused
                                 preprocessing, the Triton path, the wheel's own CUDA call
    "cuda-shipped", "cuda-tuned" on, with one of the wheel's two CUDA kernels forced (developers, below): the
                                 shipped one (= "wheel"), the round J tuned one (no timing at first use)
    anything else                off, with a warning

Where it applies: CUDA tensors in bf16, no autograd, no autocast, head size 128, the full self-attention of ONE
sequence (sparse: q, k, v sparse tensors of one prop with the same length, the call with three arguments; dense:
q, k, v [B, L, H, 128] with as many keys as queries). Cross-attention (5 context tokens), several sequences and
every other call keep their backend.

The kernel: on sm_8x (RTX 30xx / 40xx, A100) sageattn_qk_int8_pv_fp16_cuda with qk_quant_gran "per_warp" and
pv_accum_dtype "fp16+fp32" (woct0rdho's post3 carries SASS for sm_80 only, which then runs on every 8.x card;
Spellcast3D's own builds from post4 on carry SASS for sm_80, sm_86 and sm_89 - the driver runs the card's own -
and PTX for compute_89), except at the token counts TRITON_TOKENS lists for the installed wheel and the card's
architecture: there sage_triton.triton runs - bit for bit the answer
of the wheel's sageattn_qk_int8_pv_fp16_triton (proven with torch.equal on the flows' real tensors), with its
preprocessing in one Triton pass and its kernel compiled without Triton's length specialisation (which made it
7-10 % slower at lengths divisible by 16). Both kernels quantise q and k to int8 (the CUDA kernel q per 16 tokens,
the Triton one per 128) and sum P.V over 64 keys in fp16 before adding to fp32: they differ in the last bits, not
in quality (round I: the attention error 1.01-1.17 times the CUDA kernel's; round J: whole hr / tex samplings at
1536 end 1.03 / 0.84 times as far from the CUDA kernel's as the largest accepted yardstick does). The table is a
measurement (code/devtools/pixal3d-lowmem/round-j/sagefast/sj_rule.py in Spellcast3D prints it for a card and a
wheel): a fixed rule rather than a timing at first use, so that a prop gives the same numbers in every run (the
two CUDA kernels below give the same bits, so between those a timing at first use decides).
On sm_90 and newer, whose cards cannot run that SASS, sageattn_qk_int8_pv_fp16_triton. Both cast v to fp16: v
is therefore scaled by 2^-V_SHIFT before the call and the result by 2^V_SHIFT after it - exact in bf16, the same
values to the last digit (measured) - so that neither overflows below |v| ~ 262,000 (the flows' largest |v|
measured: 860; without the scaling the fp16 partial sums could overflow from |v| ~ 1,023, and the pure-fp16
accumulator, "fp16", gives Inf on the flows' real tensors); the engine folds that scaling into the weights (`fold`).
Sequences of any length are handled by the kernels (masked last block; lengths of 64 tokens or fewer are not
taken: the wheel's sm_80 kernel reads unset shared memory there, fixed upstream after the build we use).

Two CUDA kernels, the same bits (sm_8x; Spellcast3D's wheel from post5 on): the wheel holds the kernel as shipped
(the op qk_int8_sv_f16_accum_f16_attn_inst_buf, which sageattn_qk_int8_pv_fp16_cuda calls) and round J's tuned
build of the same kernel (TUNED_OP, the same arguments: two blocks an SM, the P.V step one output tile at a time;
-13 % a call on an RTX 3060, unmeasured on other cards - it may be slower on some, which must never happen to a
user). Both run the same arithmetic in the same order: the same answer bit for bit (torch.equal on the flows'
captured q / k / v and on random tables, round J sagecuda), so the choice between them can never change a result,
only the time. The choice is made per process on the card itself: at the self-check, after both have given the
same bits on its table, `_race` times the two ops on the same random int8 tables (RACE_TOKENS: 21,504 and 8,192
tokens, 12 heads x 128, interleaved ABBA, CUDA events, about half a second on an RTX 3060, the tables freed after)
and the tuned one serves only where it is at least RACE_MARGIN (5 %) faster at every size timed - otherwise, and
wherever the timing cannot run (an older wheel without the op, no memory for the tables, an error), the shipped
one, as before.
One `[SAGE]` line says which won and by how much; PIXAL3D_SAGE=cuda-shipped / cuda-tuned skips the timing.

Before the first SageAttention call of a process each kernel the card may use runs once on a small random table
against an fp32 reference (`_check`; the Triton kernels compile there); a kernel that fails or answers wrongly is
left out (the CUDA kernel then serves every length), and a card older than sm_80, a missing or broken wheel or no
kernel left turns SageAttention off with a `[SAGE]` warning, and the flows keep flash_attn.
"""

from __future__ import annotations

import os
import statistics
import time
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
_KERNELS = {"cuda", "triton", "wheel", "cuda-shipped", "cuda-tuned"}     # on, with the kernel forced (docstring)
_ALIASES = {"cuda-shipped": "wheel", "cuda-tuned": "tuned"}             # PIXAL3D_SAGE spelling -> candidate name
TUNED_OP = "qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d"    # torch.ops.sageattention_qattn_sm80.<this>: the tuned
#                     kernel (Spellcast3D's wheel post5 on; the shipped op's arguments, the same bits)
RACE_TOKENS = ((21504, 4), (8192, 6))   # the shipped and the tuned op timed at first use: (tokens, calls of each)
#                     - an hr / tex pass's size and a smaller one; 12 heads x 128, interleaved ABBA (an even count:
#                     drift weighs on both alike), the median; ~0.5 s on an RTX 3060
RACE_MARGIN = 0.05  # the tuned op serves only where it is at least this much faster at every size timed. Round J on
#                     an RTX 3060: the call-by-call race reads the tuned op ~1 point better than the flows' passes do
#                     (-12.8 % against -11.8 % a call at 21.5k tokens; blocks of one kernel: -10.5 %), and a median of
#                     a few calls varies by about +-3 points - 5 % keeps the shipped op on a card where the tuned one
#                     is not clearly faster, and loses nothing on the 3060 (the race there: -9 to -15 %)
# sm_8x, the automatic choice (PIXAL3D_SAGE on or unset), measured per card architecture (round J, sagefast):
FUSED_PREP = False  # the CUDA kernel fed by sage_triton.ship_prep (one Triton pass) instead of the wheel's own
#                     preprocessing; taken only where the self-check finds the wheel's bits on its table
TRITON_TOKENS: dict = {     # (SageAttention version, major, minor) -> ((lo, hi), ...): token counts lo <= N < hi
    #                          where sage_triton.triton (the wheel's Triton kernel, unspecialised, its preprocessing
    #                          fused) runs instead of the CUDA kernel. Measured per wheel and card architecture
    #                          (round J, sagefast: sj_micro.py / sj_rule.py); a wheel or architecture not listed
    #                          keeps the CUDA kernel at every N until it is measured.
    # post3 (woct0rdho's build) on sm_86 (RTX 3060): the Triton path 7-9 % faster a call from 13,307 to 49,152
    # tokens (-8 % at 21,547), an hr / tex pass at 1536 -141 / -170 ms, whole samplings within the yardsticks'
    # class; from 8,192 tokens only: the detail flows (ss and lr, 2-4k tokens, keep the CUDA kernel - their whole
    # samplings were not re-checked with it). Round J's own build post4 (a faster CUDA kernel, the same bits):
    # the CUDA kernel 4-6 % faster than the Triton path at every N from 1,216 to 52,965 - no entry. post5 (post4's
    # tuned kernel beside the shipped one, chosen by `_race`): no entry either.
    ("2.2.0+cu128torch2.8.0.post3", 8, 6): ((8192, 1 << 30),),
}


def from_env() -> bool:
    """PIXAL3D_SAGE as a bool (the config modules call this once, when they are imported)."""
    text = os.environ.get("PIXAL3D_SAGE", "").strip().lower()
    if text == "":
        return DEFAULT
    if text in _ON or text in _KERNELS:
        return True
    if text not in _OFF and text not in _warned:
        _warned.add(text)
        print(f"[SAGE] PIXAL3D_SAGE={text!r} is none of on, off, cuda, triton, wheel, cuda-shipped, cuda-tuned: "
              f"SageAttention stays off", flush=True)
    return False


def preference() -> str:
    """The kernel PIXAL3D_SAGE asks for: "cuda", "triton", "wheel" (also for cuda-shipped), "tuned" (cuda-tuned),
    or "auto" (on / unset)."""
    text = os.environ.get("PIXAL3D_SAGE", "").strip().lower()
    return _ALIASES.get(text, text) if text in _KERNELS else "auto"


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


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaled: bool = False) -> torch.Tensor:
    """Softmax(q k^T / sqrt(D)) v for q, k, v [B, N, H, D] (last dimension contiguous), not causal: [B, N, H, D] in
    q's dtype. Call only where usable() said True. `scaled`: v already carries the 2^-V_SHIFT (the weights were
    folded, `fold`): the answer keeps it, and the folded to_out undoes it."""
    if scaled:
        return _kernel(q, k, v)
    out = _kernel(q, k, v * 2.0 ** -V_SHIFT)
    return out.mul_(2.0 ** V_SHIFT)


def fold(model: torch.nn.Module) -> int:
    """The 2^-V_SHIFT of v and the 2^V_SHIFT of the answer folded into the weights of every full self-attention of
    `model`: to_qkv's v rows (weight and bias) times 2^-V_SHIFT, to_out's weight times 2^V_SHIFT. Powers of two
    scale a floating-point value exactly (no value of the flows comes near the exponent's ends), and every product
    and sum of the attention and of the two GEMMs scales with them, so every path - SageAttention without its two
    scaling passes (0.8 ms a block at 21.5k tokens), flash_attn, SDPA, the int8 layers (their row and channel
    scales carry the powers) - gives the same bits as before. Done once per loaded network (the module's
    `sage_folded` says so), by the engine when the switch is on. Returns how many modules were folded."""
    down, up = 2.0 ** -V_SHIFT, 2.0 ** V_SHIFT
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if (type(module).__name__ not in ("SparseMultiHeadAttention", "MultiHeadAttention")
                    or getattr(module, "_type", None) != "self" or getattr(module, "attn_mode", None) != "full"
                    or getattr(module, "sage_folded", False)):
                continue
            C = module.channels
            _scale_rows(module.to_qkv, slice(2 * C, 3 * C), down)
            _scale_rows(module.to_out, slice(0, C), up, bias=False)
            module.sage_folded = True
            count += 1
    return count


def _scale_rows(linear: torch.nn.Module, rows: slice, factor: float, bias: bool = True) -> None:
    """The output rows `rows` of a linear layer (nn.Linear, or an int8 layer with `weight_q` and `scale`) times
    `factor`: the weight rows, or the int8 layer's channel scales, and the bias rows when `bias`."""
    if hasattr(linear, "weight_q"):
        linear.scale[rows].mul_(factor)
    else:
        linear.weight[rows].mul_(factor)
    if bias and getattr(linear, "bias", None) is not None:
        linear.bias[rows].mul_(factor)


def _candidates(device: torch.device, want: str) -> dict[str, tuple[Callable, str]]:
    """The kernels to try on this card for the preference `want` (preference()): {name: (kernel(q, k, v), label)},
    each [B, N, H, 128] NHD bf16 -> the same. "wheel" always; on sm_8x "tuned" where the wheel has TUNED_OP and
    the choice is automatic or asks for it, "cuda" where FUSED_PREP or asked for, and "triton" where TRITON_TOKENS
    lists this architecture or it is asked for (a kernel not needed is not compiled)."""
    import sageattention as sa

    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError(f"the card is sm_{major}{minor}; SageAttention's int8 kernels are used from sm_80 on")

    def wheel_cuda(q, k, v):
        return sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD", is_causal=False,
                                                qk_quant_gran="per_warp", pv_accum_dtype="fp16+fp32")

    def wheel_triton(q, k, v):
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD", is_causal=False,
                                                  quantization_backend="triton")

    if major != 8:
        return {"wheel": (wheel_triton, "sageattn_qk_int8_pv_fp16_triton")}
    found = {"wheel": (wheel_cuda, "sageattn_qk_int8_pv_fp16_cuda (per_warp, fp16+fp32)")}
    op = _tuned_op() if want in ("auto", "tuned") else None
    if op is not None:
        found["tuned"] = (_wheel_call(op), f"sageattn_qk_int8_pv_fp16_cuda (per_warp, fp16+fp32), the tuned kernel "
                                           f"({TUNED_OP})")
    if want in ("wheel", "tuned"):
        return found
    from . import sage_triton

    if want == "cuda" or (want == "auto" and FUSED_PREP):
        found["cuda"] = (sage_triton.ship, "the wheel's sm80 kernel (per_warp, fp16+fp32), preprocessing fused")
    if want == "triton" or (want == "auto" and _triton_ranges(device)):
        found["triton"] = (sage_triton.triton, "the wheel's Triton kernel, unspecialised, preprocessing fused")
    return found


def _tuned_op() -> Optional[Callable]:
    """The wheel's tuned CUDA op (TUNED_OP), or None for a wheel without it (woct0rdho's post3, Spellcast3D's post4)."""
    import sageattention.core  # noqa: F401  (loads _qattn_sm80, which registers torch.ops.sageattention_qattn_sm80)

    try:
        return getattr(torch.ops.sageattention_qattn_sm80, TUNED_OP)
    except (AttributeError, RuntimeError):
        return None


def _wheel_call(op: Callable) -> Callable:
    """sageattn_qk_int8_pv_fp16_cuda(q, k, v, "NHD", is_causal=False, qk_quant_gran="per_warp", pv_accum_dtype=
    "fp16+fp32") step for step as the wheel's core.py takes it at head size 128 (no padding, k's mean subtracted,
    no LSE) - the same functions on the same tensors in the same order - with `op` as its CUDA kernel."""
    from sageattention.quant import per_warp_int8

    def run(q, k, v):
        torch.cuda.set_device(v.device)
        km = k.mean(dim=1, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=16, BLKK=64)
        o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
        op(q_int8, k_int8, v.to(torch.float16), o, q_scale, k_scale, 0, 0, 2, q.size(-1) ** -0.5, 0)
        return o

    return run


def _race(device: torch.device) -> tuple[bool, str]:
    """The shipped and the tuned CUDA op timed on this card (see the docstring): the same random int8 q (also as k)
    and fp16 v at each of RACE_TOKENS, 12 heads x 128, the calls interleaved (shipped, tuned / tuned, shipped), one
    CUDA event pair each, the medians compared (round J: the tuned / shipped ratio on these tables is the one on
    the bench's quantised tables and on the flows' captured q / k / v to ~1 point, race_check.py). Both ops have run on the self-check's table just before (loaded,
    the same bits). Returns (the tuned op serves, the [SAGE] line). Raises on a CUDA error; a size whose tables
    (5 bytes a token and channel) do not fit in what the card and PyTorch's cache have free is left out."""
    ops = torch.ops.sageattention_qattn_sm80
    fns = {"shipped": ops.qk_int8_sv_f16_accum_f16_attn_inst_buf, "tuned": getattr(ops, TUNED_OP)}
    heads, began = 12, time.perf_counter()
    rows, skipped = [], []
    torch.cuda.synchronize(device)
    for n, calls in RACE_TOKENS:
        need = n * heads * HEAD_DIM * 5                  # int8 q (= k), fp16 v, bf16 output
        free = torch.cuda.mem_get_info(device)[0]
        spare = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if need * 1.25 > free + spare:
            skipped.append(n)
            continue
        g = torch.Generator(device=device).manual_seed(n)
        q8 = torch.randint(-127, 128, (1, n, heads, HEAD_DIM), dtype=torch.int8, device=device, generator=g)
        v16 = torch.randn((1, n, heads, HEAD_DIM), dtype=torch.float16, device=device, generator=g)
        o = torch.empty((1, n, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
        q_scale = torch.full((1, heads, (n + 127) // 128 * 8), 0.02, dtype=torch.float32, device=device)
        k_scale = torch.full((1, heads, (n + 63) // 64), 0.02, dtype=torch.float32, device=device)
        events: dict = {name: [] for name in fns}
        stream = torch.cuda.default_stream(device)       # the ops launch on the legacy default stream
        for i in range(calls):
            for name in (("shipped", "tuned") if i % 2 == 0 else ("tuned", "shipped")):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                fns[name](q8, q8, v16, o, q_scale, k_scale, 0, 0, 2, HEAD_DIM ** -0.5, 0)
                end.record(stream)
                events[name].append((start, end))
        torch.cuda.synchronize(device)
        ms = {name: statistics.median(a.elapsed_time(b) for a, b in pairs) for name, pairs in events.items()}
        rows.append((n, ms["shipped"], ms["tuned"]))
        del q8, v16, o, q_scale, k_scale, events
    seconds = time.perf_counter() - began
    if not rows:
        return False, (f"[SAGE] the shipped CUDA kernel serves: the tuned one was not timed (no room for the "
                       f"tables of {', '.join(f'{n:,}' for n in skipped)} tokens)")
    wins = all(t <= s * (1 - RACE_MARGIN) for _, s, t in rows)
    text = ", ".join(f"{n:,} tokens {s:.2f} -> {t:.2f} ms ({100 * (t / s - 1):+.1f} %)" for n, s, t in rows)
    if skipped:
        text += f"; {', '.join(f'{n:,}' for n in skipped)} not timed (no room)"
    if wins:
        return True, (f"[SAGE] the tuned CUDA kernel (round J) serves, timed against the shipped one: {text} "
                      f"({seconds:.2f} s)")
    return False, (f"[SAGE] the shipped CUDA kernel serves: the tuned one (round J) is not {100 * RACE_MARGIN:.0f} % "
                   f"faster at every size timed: {text} ({seconds:.2f} s)")


def _triton_ranges(device: torch.device) -> tuple:
    """TRITON_TOKENS' ranges for the installed wheel and this card's architecture (none: the CUDA kernel serves)."""
    from importlib import metadata

    try:
        version = metadata.version("sageattention")
    except Exception:        # noqa: BLE001 - no distribution metadata: nothing was measured for it
        return ()
    return tuple(TRITON_TOKENS.get((version, *torch.cuda.get_device_capability(device)), ()))


def _by_length(cuda: Callable, triton: Callable, ranges: tuple) -> Callable:
    def run(q, k, v):
        n = q.shape[1]
        for lo, hi in ranges:
            if lo <= n < hi:
                return triton(q, k, v)
        return cuda(q, k, v)
    return run


def _choose(device: torch.device, cands: dict, tried: dict[str, Optional[torch.Tensor]]) -> tuple[Callable, str]:
    """The kernel for this card from the candidates that passed the self-check (`tried`: name -> answer on the
    check table, None for one that failed). sm_8x: the CUDA kernel - the tuned op with the wheel's preprocessing
    ("tuned") where it gives the shipped op's bits on the table and wins `_race`, else the fused preprocessing
    ("cuda") where it gives the wheel's own bits on the table, else the wheel's own call - and the Triton path at
    the token counts TRITON_TOKENS lists for this architecture; PIXAL3D_SAGE=cuda / triton / wheel / cuda-shipped /
    cuda-tuned forces one. sm_90 and newer: the wheel's Triton call."""
    ok = {name: cands[name] for name, got in tried.items() if got is not None}
    want = preference()
    if "cuda" in ok and "wheel" in ok and not torch.equal(tried["cuda"], tried["wheel"]):
        print("[SAGE] the fused preprocessing does not give the wheel's bits on this card: the wheel's own call "
              "serves", flush=True)
        del ok["cuda"]
    if "tuned" in ok and ("wheel" not in ok or not torch.equal(tried["tuned"], tried["wheel"])):
        print("[SAGE] the tuned CUDA kernel does not give the shipped kernel's bits on this card: the shipped one "
              "serves", flush=True)
        del ok["tuned"]
    if want in ok:
        return ok[want]
    if want != "auto":
        print(f"[SAGE] PIXAL3D_SAGE={os.environ.get('PIXAL3D_SAGE', want).strip().lower()}: that kernel is not "
              f"available on this card or failed its self-check; the automatic choice serves", flush=True)
    cuda = ok.get("cuda") or ok.get("wheel")
    if "tuned" in ok:
        try:
            wins, line = _race(device)
        except Exception as exc:        # noqa: BLE001 - a timing that fails changes nothing: the shipped op serves
            wins, line = False, (f"[SAGE] the shipped CUDA kernel serves: timing the tuned one failed "
                                 f"({type(exc).__name__}: {str(exc)[:200]})")
        print(line, flush=True)
        if wins:
            cuda = ok["tuned"]
    triton = ok.get("triton")
    if cuda is None:
        raise RuntimeError("the CUDA kernel did not pass its self-check")
    ranges = _triton_ranges(device)
    if triton is None or not ranges:
        return cuda
    text = ", ".join(f"{lo}+" if hi >= 1 << 30 else f"{lo}-{hi - 1}" for lo, hi in ranges)
    return _by_length(cuda[0], triton[0], ranges), f"{cuda[1]}; at {text} tokens {triton[1]}"


def _check(device: torch.device) -> bool:
    """Every candidate kernel once on a random table (1000 tokens: more than one block of keys and not a multiple
    of it; v as a strided view and 1000 times larger than q and k, as the flows' v is), against fp32 attention:
    finite, and within 3 % relative L2 (they measure ~1.3 % there, flash_attn ~0.23 %); a candidate that fails or
    raises is left out, and _choose picks among the others (on sm_8x timing the two CUDA kernels, `_race`, when
    both are there). None left turns SageAttention off."""
    global _state, _kernel, KERNEL
    try:
        cands = _candidates(device, preference())
        g = torch.Generator(device=device).manual_seed(0)
        n, heads = 1000, 12
        qkv = torch.randn((1, n, 3, heads, HEAD_DIM), device=device, dtype=torch.bfloat16, generator=g)
        qkv[:, :, 2] *= 1000
        q, k, v = qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2]
        want_out = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2),
                                                  v.float().transpose(1, 2)).transpose(1, 2)
        vs = v * 2.0 ** -V_SHIFT
        tried: dict[str, Optional[torch.Tensor]] = {}
        notes = []
        for name in cands:
            try:
                got = cands[name][0](q, k, vs)
                torch.cuda.synchronize(device)
                rel = float((got.float() * 2.0 ** V_SHIFT - want_out).norm() / want_out.norm())
                if got.shape != want_out.shape or not bool(torch.isfinite(got).all()) or not rel <= 0.03:
                    raise RuntimeError(f"its answer differs from fp32 attention (relative L2 {rel:.3g}, shape "
                                       f"{tuple(got.shape)})")
                tried[name] = got
            except Exception as exc:        # noqa: BLE001 - whatever the wheel, the card or Triton raises
                tried[name] = None
                notes.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
        if notes:
            print(f"[SAGE] left out after the self-check: {'; '.join(notes)}", flush=True)
        _kernel, KERNEL = _choose(device, cands, tried)
    except Exception as exc:        # noqa: BLE001 - whatever the wheel, the card or Triton raises: flash_attn serves
        _state, _kernel = False, None
        print(f"[SAGE] SageAttention does not run on this card ({type(exc).__name__}: {str(exc)[:300]}); the "
              f"flows keep flash_attn", flush=True)
        return False
    _state = True
    print(f"[SAGE] the flows' full self-attention runs SageAttention: {KERNEL} (PIXAL3D_SAGE)", flush=True)
    return True
