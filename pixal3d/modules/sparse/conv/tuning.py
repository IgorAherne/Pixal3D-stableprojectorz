"""flex_gemm's masked implicit GEMM (the decoders' sparse convolutions): its tile configuration chosen by a steadier
measurement when a key is tuned, every key tuned once at install, and no key tuned again while the artist works.

flex_gemm tunes the kernel through Triton's autotuner, once per key (log2 of the rows, Ci, Co) per install, the first
time the key occurs, and keeps the choice in its cache file. Triton's benchmark sizes its runs from a 5-call estimate
(do_bench: 25 ms of warm-up, 100 ms of repetitions), so for a convolution of 50-80 ms it times a single run of each
configuration, with whatever else is on the card and the clocks wherever the compile pause before it left them. On an
RTX 3060 that kept, for the 512x512 convolutions of the 1536 props' level 1 (34 calls a prop), a configuration that
takes 78.6 ms where another of flex_gemm's own takes 51.9 ms (Spellcast3D code/devtools/pixal3d-lowmem/round-j/parts:
conv_one.py, retune_masked.py).

`install()` gives the kernel's autotuner `steady_bench`: the configuration's calls are repeated until the card has
been busy for STEADY_WARM_MS, then the median of STEADY_RUNS timed calls is taken (3 for a call above 20 ms: about
as many calls as Triton's own benchmark makes of a long kernel, where it times a single one); a configuration whose
first steady call already takes EARLY_OUT times the best median of the key so far is not timed further (it cannot be
chosen; most of a key's configurations are such). EXTRA_CONFIGS adds tiles flex_gemm's list lacks. Every configuration gives the same bits (SPLITK 1: each output row sums its own neighbours' offsets in
ascending order, BK channels at a time), so neither changes a value: only which tiles do the work. The split-K
wrapper's own tuner (SPLITK, which does change the sums) keeps its own benchmark.

ENABLED = False: Triton's own benchmark, flex_gemm's own list.

Calibrated once, at install (`calibrate`, which `python -m pixal3d.calibrate` runs; the work is in calibrate_conv.py):
every size the decoders can meet is tuned on a quiet card through the functions the decoders call and saved in
flex_gemm's cache file, this card's earlier entries dropped first; the grid of sizes is kept in the install's
calibration record (pixal3d/calibration.py, section "flex_gemm"). Where that record exists for this card, `install()`
also gives the three tuners of the masked convolution - the split-K wrapper's (SPLITK), the masked kernel's and the
split kernel's (tiles) - a lookup that tunes nothing (NEAREST, NearestCache): a key the tuner has not got takes the
configuration of the nearest calibrated key of the same shape (Ci, Co, V and the key's other parts; the nearest log2
of the rows) instead of being measured while the artist waits. The masked kernel's tile never changes a value; the
SPLITK and the split kernel's tile do change the last bits of a decode, as flex_gemm's own per-install choice always
has. Without a record (an uncalibrated install, another card, the golden checks) flex_gemm tunes what it meets, as
before.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import os
import re
import statistics
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

ENABLED = True
STEADY_WARM_MS = 60.0       # the card busy with the configuration's own calls before it is timed
STEADY_RUNS = 5             # timed calls; their median decides (STEADY_RUNS_LONG for calls above LONG_MS)
STEADY_RUNS_LONG = 3
LONG_MS = 20.0
EARLY_OUT = 1.25            # a configuration this much slower than the key's best so far: its first steady call decides
# (B1, B2, BK, num_warps, num_stages) beside flex_gemm's list: on an RTX 3060 the fastest tile of most of the
# decoders' keys from 2^16 rows on (128x128x64: the 512x512 convolutions of level 1 43.8 ms against the cached 76.6,
# 256x1024 x1.40, the rest +1-5 %) and of every 64x64 key (64x64x64: x1.22-1.44); neither is in flex_gemm's list.
# A card that cannot hold a tile's shared memory gets "out of resources" for it, which the tuner skips.
EXTRA_CONFIGS: Sequence[tuple] = ((128, 128, 64, 4, 3), (64, 64, 64, 4, 3))
KERNEL = "sparse_submanifold_conv_fwd_masked_implicit_gemm"
# the three tuners of the masked convolution: (module under flex_gemm.kernels.triton.spconv, attribute)
TUNERS = {"masked": (KERNEL, f"{KERNEL}_kernel"), "split": (f"{KERNEL}_splitk", f"{KERNEL}_splitk_kernel"),
          "splitk": (f"{KERNEL}_splitk", f"{KERNEL}_splitk")}
NEAREST = True              # with this card's calibration record: a missing key takes the nearest calibrated choice
BORROWED: List[Tuple[str, str, str]] = []   # (tuner, key, the calibrated key whose choice it took), this process
_installed = False
_nearest_installed = False
_suspended = 0              # > 0 while calibrate() tunes: every missing key is measured
_grid_cache: Dict[str, Any] = {}
_key_best = [float("inf")]  # the best median of the key being tuned (reset when the tuner starts a key)


def steady_bench(kernel_call: Callable[[], None], quantiles: Optional[Sequence[float]] = None) -> List[float]:
    """The time of `kernel_call` (ms) as Triton's autotuner asks for it ([median, p20, p80] with `quantiles`): first
    call (compiles), calls until the card has worked STEADY_WARM_MS on them (at least one), then the median of
    STEADY_RUNS timed calls (STEADY_RUNS_LONG when a call takes more than LONG_MS: about as many calls as Triton's
    own benchmark makes of a long kernel, 5 + 1 + 1). A configuration whose first steady call takes more than
    EARLY_OUT x the key's best median so far returns that call's time (it is not chosen)."""
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
        if EARLY_OUT and busy >= STEADY_WARM_MS and last > EARLY_OUT * _key_best[0]:
            return last if quantiles is None else [last for _ in quantiles]
    runs = STEADY_RUNS_LONG if last > LONG_MS else STEADY_RUNS
    times = sorted(timed() for _ in range(runs))
    median = statistics.median(times)
    _key_best[0] = min(_key_best[0], median)
    if quantiles is None:
        return median
    return [median if q == 0.5 else times[min(len(times) - 1, int(q * len(times)))] for q in quantiles]


def install() -> bool:
    """Give flex_gemm's masked kernel autotuner the steady benchmark and EXTRA_CONFIGS, and the three convolution
    tuners the calibrated lookup (NEAREST; it acts only where this card's calibration record exists), once per
    process. False when the steady tuning is off or when flex_gemm is not the one this was written for (its
    autotuner then works as before)."""
    global _installed
    if NEAREST:
        _install_nearest()
    if _installed or not ENABLED:
        return _installed
    try:
        import triton
        tuner = getattr(importlib.import_module(f"flex_gemm.kernels.triton.spconv.{KERNEL}"), f"{KERNEL}_kernel")
        if not hasattr(tuner, "configs") or not hasattr(tuner, "cache"):
            return False
        tuner.__dict__["do_bench"] = steady_bench          # shadows Triton's cached property on this instance
        prune = tuner.prune_configs                         # called once per key, before its configurations are timed

        def pruned(*args: Any, **kwargs: Any) -> Any:
            _key_best[0] = float("inf")
            return prune(*args, **kwargs)

        tuner.prune_configs = pruned
        have = {(c.kwargs.get("B1"), c.kwargs.get("B2"), c.kwargs.get("BK"), c.num_warps, c.num_stages)
                for c in tuner.configs}
        extra = [triton.Config({"B1": b1, "B2": b2, "BK": bk}, num_warps=warps, num_stages=stages)
                 for b1, b2, bk, warps, stages in EXTRA_CONFIGS if (b1, b2, bk, warps, stages) not in have]
        tuner.configs[:0] = extra       # first: the usual winners set the bar EARLY_OUT holds the others to
        _installed = True
    except Exception as exc:        # noqa: BLE001 - tuning as before
        print(f"[SPARSE] the convolutions' steady tuning is off ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        return False
    return True


# ------------------------------------------------------------------------------------------------ the calibrated lookup

def tuners() -> Dict[str, Any]:
    """flex_gemm's tuners of the masked convolution that this flex_gemm has: {"masked", "split", "splitk"} (the
    masked kernel's and the split kernel's Triton autotuners, the split-K wrapper's tuner)."""
    found = {}
    for name, (module, attr) in TUNERS.items():
        try:
            tuner = getattr(importlib.import_module(f"flex_gemm.kernels.triton.spconv.{module}"), attr)
        except (ImportError, AttributeError):
            continue
        if isinstance(getattr(tuner, "cache", None), dict):
            found[name] = tuner
    return found


_WRAPPER_KEY = re.compile(r"^\(2\^(\d+), (\d+), (\d+), (\d+)\)$")


def parse_key(name: str, key: str) -> Optional[Tuple[int, tuple]]:
    """(log2 rows, the rest of the key) of a tuner's cache key, or None when it is not one this knows: the split-K
    wrapper's "(2^logN, Ci, Co, V)", the Triton tuners' str of (LOGN, Ci, Co, V, [SPLITK,] allow_tf32, dtypes...)."""
    if name == "splitk":
        m = _WRAPPER_KEY.match(key)
        if not m:
            return None
        logn, ci, co, v = (int(g) for g in m.groups())
        return logn, (ci, co, v)
    try:
        parts = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return None
    if not (isinstance(parts, tuple) and len(parts) >= 4 and all(type(p) is int for p in parts[:4])):
        return None
    return parts[0], tuple(parts[1:])


def _grid() -> Optional[Dict[Tuple[int, int, int], frozenset]]:
    """The calibrated sizes {(Ci, Co, V): log2 rows tuned} of this install's record for this card - its "keys" (the
    sizes tuned; one skipped for memory is not among them), else its "grid" spans - or None (no record, another
    card's, a failed calibration step, NEAREST off, or calibrate() at work)."""
    if _suspended or not NEAREST:
        return None
    if "grid" not in _grid_cache:
        grid: Optional[dict] = None
        try:
            from .... import calibration

            section = calibration.section("flex_gemm")
            if section and section.get("ok", True) and isinstance(section.get("grid"), dict):
                grid = {}
                if isinstance(section.get("keys"), list):
                    for row in section["keys"]:
                        ci, co, v = (int(x) for x in str(row[0]).split("x"))
                        grid.setdefault((ci, co, v), set()).add(int(row[1]))
                else:
                    for shape, (lo, hi) in section["grid"].items():
                        ci, co, v = (int(x) for x in shape.split("x"))
                        grid[(ci, co, v)] = set(range(int(lo), int(hi) + 1))
                grid = {shape: frozenset(sizes) for shape, sizes in grid.items()}
        except Exception as exc:        # noqa: BLE001 - an unreadable record: flex_gemm tunes as before
            print(f"[SPARSE] the convolutions' calibration record is not used ({type(exc).__name__}: "
                  f"{str(exc)[:120]})", flush=True)
            grid = None
        _grid_cache["grid"] = grid or None
    return _grid_cache["grid"]


def forget_grid() -> None:
    """Read the calibration record again at the next missing key (a test; the record was written meanwhile)."""
    _grid_cache.clear()


class NearestCache(dict):
    """A tuner's cache (flex_gemm's dict, key -> configuration) whose membership test - the tuners' `key not in
    self.cache`, chunked.reference_splitk's `key in cache` - gives a key it has not got the configuration of the
    nearest calibrated key of the same shape (`nearest`), so the tuner runs that configuration instead of measuring.
    Without this card's calibration record it is flex_gemm's dict."""

    def __init__(self, data: dict, name: str):
        super().__init__(data)
        self.name = name

    def __contains__(self, key: object) -> bool:
        if dict.__contains__(self, key):
            return True
        if not isinstance(key, str):
            return False
        found = nearest(self, self.name, key)
        if found is None:
            return False
        source, config = found
        dict.__setitem__(self, key, dict(config) if isinstance(config, dict) else config)
        BORROWED.append((self.name, key, source))
        if os.environ.get("FLEX_GEMM_AUTOTUNER_VERBOSE", "0") == "1":
            print(f"[SPARSE] {self.name} {key}: the calibrated choice of {source} (not tuned)", flush=True)
        return True


def nearest(cache: dict, name: str, key: str) -> Optional[Tuple[str, Any]]:
    """(the calibrated key, its configuration) nearest to `key` among `cache`'s: the same shape and other key parts,
    log2 rows the record says were calibrated for that shape (an entry borrowed earlier - in this process, or saved
    into the file by a later save - is never a source), the nearest log2 rows (the larger of two as near); None when
    there is no record or no such key (the tuner then measures, as flex_gemm does)."""
    grid = _grid()
    if not grid:
        return None
    parsed = parse_key(name, key)
    if parsed is None:
        return None
    logn, rest = parsed
    sizes = grid.get(tuple(rest[:3]))
    if not sizes:
        return None
    best = None
    for other in list(dict.keys(cache)):
        got = parse_key(name, other)
        if got is None or got[1] != rest or got[0] not in sizes:
            continue
        rank = (abs(got[0] - logn), -got[0])
        if best is None or rank < best[0]:
            best = (rank, other)
    if best is None:
        return None
    return best[1], dict.__getitem__(cache, best[1])


def _install_nearest() -> bool:
    """Give the three tuners a NearestCache (their entries kept). The split-K wrapper's cache is put back into one
    at each call should a load_autotune_cache have replaced it (flex_gemm gives that tuner a new dict there; the
    Triton tuners' caches are filled in place)."""
    global _nearest_installed
    if _nearest_installed:
        return True
    try:
        found = tuners()
        for name, tuner in found.items():
            if not isinstance(tuner.cache, NearestCache):
                tuner.cache = NearestCache(tuner.cache, name)
        wrapper = found.get("splitk")
        if wrapper is not None and callable(getattr(wrapper, "key_fn", None)):
            key_fn = wrapper.key_fn

            def keyed(*args: Any, **kwargs: Any) -> Any:
                if not isinstance(wrapper.cache, NearestCache):
                    wrapper.cache = NearestCache(wrapper.cache, "splitk")
                return key_fn(*args, **kwargs)

            wrapper.key_fn = keyed
        _nearest_installed = bool(found)
    except Exception as exc:        # noqa: BLE001 - flex_gemm tunes as before
        print(f"[SPARSE] the convolutions' calibrated lookup is off ({type(exc).__name__}: {str(exc)[:120]})",
              flush=True)
    return _nearest_installed


@contextlib.contextmanager
def suspended() -> Iterator[None]:
    """No calibrated lookup inside: every missing key is tuned (calibrate)."""
    global _suspended
    _suspended += 1
    try:
        yield
    finally:
        _suspended -= 1


def _tile(config: Any) -> Optional[str]:
    kw = getattr(config, "kwargs", None) or {}
    return f"{kw.get('B1')}x{kw.get('B2')}x{kw.get('BK')}/{config.num_warps}/{config.num_stages}" if kw else None


def choice(logn: int, ci: int, co: int, v: int = 27) -> dict:
    """What the tuners hold for a float16 convolution of 2^logn rows: {"splitk", "tile" (the masked kernel's, SPLITK
    1), "split_tile" (the split kernel's at that SPLITK)}, tiles as B1xB2xBK/warps/stages, None where none is held."""
    found = tuners()
    out: Dict[str, Any] = {"splitk": None, "tile": None, "split_tile": None}
    wrapper = found.get("splitk")
    if wrapper is not None:
        got = dict.get(wrapper.cache, f"(2^{logn}, {ci}, {co}, {v})")
        out["splitk"] = int(got.get("SPLITK", 1)) if isinstance(got, dict) else None
    for name in ("masked", "split"):
        tuner = found.get(name)
        if tuner is None:
            continue
        for key in list(dict.keys(tuner.cache)):
            got = parse_key(name, key)
            if got is None or got[0] != logn or got[1][:3] != (ci, co, v) or "torch.float16" not in got[1]:
                continue
            if name == "masked":
                out["tile"] = _tile(dict.__getitem__(tuner.cache, key))
            elif out["splitk"] and out["splitk"] > 1 and got[1][3] == out["splitk"]:
                out["split_tile"] = _tile(dict.__getitem__(tuner.cache, key))
    return out


def calibrate(device: Any, say: Callable[[str], None] = print) -> dict:
    """Tune every convolution size the decoders can meet on this card, once (the install's calibration, `python -m
    pixal3d.calibrate`, on a quiet card): flex_gemm's own tuners through the functions the decoders call, with the
    steady benchmark and EXTRA_CONFIGS, saved into flex_gemm's cache file; this card's earlier entries of the three
    tuners are dropped first. Returns the record's "flex_gemm" section (calibrate_conv.run): the grid of sizes the
    calibrated lookup reads, every key's choice, time and card memory, "summary" in one line."""
    from . import calibrate_conv

    found = calibrate_conv.run(device, say=say)
    forget_grid()
    return found
