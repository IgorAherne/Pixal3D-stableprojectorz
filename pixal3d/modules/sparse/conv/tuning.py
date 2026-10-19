"""flex_gemm's masked implicit GEMM (the decoders' sparse convolutions): its tile configuration chosen by a steadier
measurement when a key is tuned.

flex_gemm tunes the kernel through Triton's autotuner, once per key (log2 of the rows, Ci, Co) per install, the first
time the key occurs, and keeps the choice in its cache file. Triton's benchmark sizes its runs from a 5-call estimate
(do_bench: 25 ms of warm-up, 100 ms of repetitions), so for a convolution of 50-80 ms it times a single run of each
configuration, with whatever else is on the card and the clocks wherever the compile pause before it left them. On an
RTX 3060 that kept, for the 512x512 convolutions of the 1536 props' level 1 (34 calls a prop), a configuration that
takes 78.6 ms where another of flex_gemm's own takes 51.9 ms (Spellcast3D code/devtools/pixal3d-lowmem/round-j/parts:
conv_one.py, retune_masked.py).

`install()` gives the kernel's autotuner `steady_bench`: the configuration's calls are repeated until the card has
been busy for STEADY_WARM_MS, then the median of STEADY_RUNS timed calls is taken (3 for a call above 20 ms: about
as many calls as Triton's own benchmark makes of a long kernel, where it times a single one). EXTRA_CONFIGS adds tiles flex_gemm's
list lacks. Every configuration gives the same bits (SPLITK 1: each output row sums its own neighbours' offsets in
ascending order, BK channels at a time), so neither changes a value: only which tiles do the work. The split-K
wrapper's own tuner (SPLITK, which does change the sums) is not touched. Choices already in the cache file stay; the
round J tool retune_masked.py measures them again.

ENABLED = False: Triton's own benchmark, flex_gemm's own list.
"""

from __future__ import annotations

import importlib
import statistics
from typing import Callable, List, Optional, Sequence

import torch

ENABLED = True
STEADY_WARM_MS = 60.0       # the card busy with the configuration's own calls before it is timed
STEADY_RUNS = 5             # timed calls; their median decides (STEADY_RUNS_LONG for calls above LONG_MS)
STEADY_RUNS_LONG = 3
LONG_MS = 20.0
# (B1, B2, BK, num_warps, num_stages) beside flex_gemm's list: on an RTX 3060 the fastest tile of most of the
# decoders' keys from 2^16 rows on (128x128x64: the 512x512 convolutions of level 1 43.8 ms against the cached 76.6,
# 256x1024 x1.40, the rest +1-5 %) and of every 64x64 key (64x64x64: x1.22-1.44); neither is in flex_gemm's list.
# A card that cannot hold a tile's shared memory gets "out of resources" for it, which the tuner skips.
EXTRA_CONFIGS: Sequence[tuple] = ((128, 128, 64, 4, 3), (64, 64, 64, 4, 3))
KERNEL = "sparse_submanifold_conv_fwd_masked_implicit_gemm"
_installed = False


def steady_bench(kernel_call: Callable[[], None], quantiles: Optional[Sequence[float]] = None) -> List[float]:
    """The time of `kernel_call` (ms) as Triton's autotuner asks for it ([median, p20, p80] with `quantiles`): first
    call (compiles), calls until the card has worked STEADY_WARM_MS on them (at least one), then the median of
    STEADY_RUNS timed calls (STEADY_RUNS_LONG when a call takes more than LONG_MS: about as many calls as Triton's
    own benchmark makes of a long kernel, 5 + 1 + 1)."""
    def timed() -> float:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        kernel_call()
        e.record()
        e.synchronize()
        return s.elapsed_time(e)

    kernel_call()
    torch.cuda.synchronize()
    busy, calls, last = 0.0, 0, 0.0
    while calls < 1 or (busy < STEADY_WARM_MS and calls < 64):
        last = timed()
        busy += last
        calls += 1
    runs = STEADY_RUNS_LONG if last > LONG_MS else STEADY_RUNS
    times = sorted(timed() for _ in range(runs))
    median = statistics.median(times)
    if quantiles is None:
        return median
    return [median if q == 0.5 else times[min(len(times) - 1, int(q * len(times)))] for q in quantiles]


def install() -> bool:
    """Give flex_gemm's masked kernel autotuner the steady benchmark and EXTRA_CONFIGS (once per process). False when
    off or when flex_gemm is not the one this was written for (its autotuner then works as before)."""
    global _installed
    if _installed or not ENABLED:
        return _installed
    try:
        import triton
        tuner = getattr(importlib.import_module(f"flex_gemm.kernels.triton.spconv.{KERNEL}"), f"{KERNEL}_kernel")
        if not hasattr(tuner, "configs") or not hasattr(tuner, "cache"):
            return False
        tuner.__dict__["do_bench"] = steady_bench          # shadows Triton's cached property on this instance
        have = {(c.kwargs.get("B1"), c.kwargs.get("B2"), c.kwargs.get("BK"), c.num_warps, c.num_stages)
                for c in tuner.configs}
        for b1, b2, bk, warps, stages in EXTRA_CONFIGS:
            if (b1, b2, bk, warps, stages) not in have:
                tuner.configs.append(triton.Config({"B1": b1, "B2": b2, "BK": bk}, num_warps=warps,
                                                   num_stages=stages))
        _installed = True
    except Exception as exc:        # noqa: BLE001 - tuning as before
        print(f"[SPARSE] the convolutions' steady tuning is off ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        return False
    return True
