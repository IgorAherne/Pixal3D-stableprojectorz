"""The install-time calibration of the decoders' sparse convolutions (tuning.calibrate's work): synthetic levels of
the sizes a prop can have, run through the same flex_gemm functions the decoders call, so that flex_gemm's own
tuners - the split-K wrapper's SPLITK and the masked kernel's tile, with tuning.py's steady benchmark and extra
tiles installed - choose and save every key once, on a quiet card, before the artist's first prop.

The keys (GRID): the decoders (shape and texture, the same for the single- and multi-view pipelines:
model_channels 1024, 512, 256, 128, 64; num_blocks 4, 16, 8, 4) run nine convolution shapes (Ci, Co), each at one
level: the ConvNeXt blocks' Ci x Ci and the up-blocks' conv1 Ci x 8 Ci' (1024x4096 at level 0, 512x2048 at 1,
256x1024 at 2, 128x512 at 3) on that level's voxels, all 3x3x3 (V 27) in float16. The rows (log2: flex_gemm's key)
of each level follow from the prop (Spellcast3D code/devtools/pixal3d-lowmem/round-j/calibrate/keys_census.py, the
golden cases sv1024 ... big1536 and this PC's cache of every prop it made): level 0 holds the upsample's 32-grid
tokens (2^9 for a small prop) and the HR tokens (up to 2^15: the token cap at 1536, 49,152), every level about 4.4x
the one before; from level 2 on a large prop is decoded in boxes (chunked.py: 2 M fine voxels a box, conv1 in
slabs), a small one whole. GRID spans each shape's buckets with one more on either side. A key outside it - a
slab's small remainder, a prop beyond the test set - takes the nearest calibrated key's choice at run time
(tuning.install's rule) instead of being tuned.

The inputs: a voxel shell (a sphere, SHELL_THICKNESS voxels thick, rows in Morton order, the first n of them) has
about the real levels' neighbourhood - 14-16 valid neighbours of 27 on average, the masked kernel's issued work
1.05-1.5x the useful one - at the bucket's geometric middle of rows (2^(logN + 1/2)), fewer where the key's tables
would take more than MAX_KEY_MIB of the card (the top buckets of the wide shapes; down to 2^logN, else the key is
skipped and its neighbour serves). Random float16 features, weights and bias.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

V = 27
# (Ci, Co): (lowest, highest log2 rows calibrated)
GRID: Dict[Tuple[int, int], Tuple[int, int]] = {
    (1024, 1024): (8, 16), (1024, 4096): (8, 16),
    (512, 512): (10, 18), (512, 2048): (10, 18),
    (256, 256): (12, 20), (256, 1024): (12, 18),
    (128, 128): (14, 20), (128, 512): (14, 20),
    (64, 64): (16, 22),
}
SHELL_THICKNESS = 1.6       # voxels: the real levels' 14-16 neighbours of 27 (calibrate/synth_stats.py)
MAX_KEY_MIB = 1500          # the card one key may take (its tables, the neighbour cache while built, the output)


def family(ci: int, co: int) -> str:
    return f"{ci}x{co}x{V}"


def max_splitk(n: int, co: int, sms: int) -> int:
    """The largest SPLITK flex_gemm's split-K wrapper offers for n rows and Co outputs on a card of `sms`
    multiprocessors (its config function: 2^i for i up to log2(32 sms / blocks of 128 x 128))."""
    blocks = -(-n // 128) * -(-co // 128)
    top = max(1, int(math.log2(32 * sms / blocks) + 1))
    return 2 ** (top - 1)


def key_mib(n: int, ci: int, co: int, sms: int) -> float:
    """MiB of the card one key's calibration takes at n rows: features, output, weight, bias, the neighbour cache
    (map, codes, order) with what its building adds (hash map, binary codes and their sort), the coords, and the
    split kernel's float32 partial sums at the largest SPLITK the wrapper offers (with their sum)."""
    tables = n * (ci + co) * 2 + co * V * ci * 2
    cache = n * (V * 4 + 4 + 8) + n * 64 + n * 16
    splitk = max_splitk(n, co, sms)
    split = (splitk + 1) * n * co * 4 if splitk > 1 else 0
    return (tables + cache + split) / 2**20


def rows_for(logn: int, ci: int, co: int, sms: int) -> Optional[int]:
    """The rows a key is calibrated at: the bucket's geometric middle, fewer where that would take more than
    MAX_KEY_MIB (down to 2^logn), None when even 2^logn would."""
    lo, hi = 1 << logn, (1 << (logn + 1)) - 1
    n = min(hi, int(round(2 ** (logn + 0.5))))
    while n > lo and key_mib(n, ci, co, sms) > MAX_KEY_MIB:
        n = max(lo, int(n * 0.9))
    return None if key_mib(n, ci, co, sms) > MAX_KEY_MIB else n


# ------------------------------------------------------------------------------------------------ synthetic levels

def _spread_bits(v: torch.Tensor) -> torch.Tensor:
    """The 21 low bits of v (int64) spread to every third bit (Morton code)."""
    v = v & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def shell(radius: float, thickness: float, device: Any) -> torch.Tensor:
    """The voxels [m, 3] int64 whose centres lie between radius - thickness/2 and radius + thickness/2 of the centre
    of a cube of side 2 * ceil(radius + thickness) + 2: column by column (x, y), the z ranges in between."""
    ro, ri = radius + thickness / 2, max(0.0, radius - thickness / 2)
    side = 2 * int(math.ceil(ro)) + 2
    c = side / 2
    ax = torch.arange(side, device=device, dtype=torch.float64) + 0.5 - c
    dx, dy = torch.meshgrid(ax, ax, indexing="ij")
    d2 = (dx * dx + dy * dy).reshape(-1)
    cols = torch.arange(side * side, device=device)
    keep = d2 < ro * ro
    d2, cols = d2[keep], cols[keep]
    zo = torch.sqrt(ro * ro - d2)                                   # |z - c| < zo
    zi = torch.sqrt(torch.clamp(ri * ri - d2, min=0.0))             # |z - c| >= zi
    # voxel k has its centre at z = k + 0.5 - c (c is a whole number: no centre at z = 0): the upper range
    # zi <= z < zo is k in [ceil(c - 0.5 + zi), ceil(c - 0.5 + zo)), the lower one -zo < z <= -zi is
    # k in [floor(c - 0.5 - zo) + 1, floor(c - 0.5 - zi) + 1); outside the hollow (zi = 0) they meet, disjoint
    up_a = torch.ceil(c - 0.5 + zi).long()
    up_b = torch.ceil(c - 0.5 + zo).long()
    lo_a = torch.floor(c - 0.5 - zo).long() + 1
    lo_b = torch.floor(c - 0.5 - zi).long() + 1
    starts = torch.cat([lo_a, up_a])
    counts = torch.cat([(lo_b - lo_a).clamp(min=0), (up_b - up_a).clamp(min=0)])
    colsx2 = torch.cat([cols, cols])
    total = int(counts.sum())
    owner = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts, output_size=total)
    first = torch.cumsum(counts, 0) - counts
    z = starts[owner] + (torch.arange(total, device=device) - first[owner])
    col = colsx2[owner]
    return torch.stack([col // side, col % side, z], dim=1)


def level_coords(n: int, device: Any, thickness: float = SHELL_THICKNESS) -> torch.Tensor:
    """n voxels [n, 4] int32 (batch 0, x, y, z) of a sphere shell `thickness` voxels thick, in Morton order: the
    first n of a shell of at least n voxels (a compact piece of it, Morton order being hierarchical)."""
    radius = max(2.0, math.sqrt(n / (4 * math.pi * thickness)))
    xyz = shell(radius, thickness, device)
    while xyz.shape[0] < n:
        radius *= 1.02
        xyz = shell(radius, thickness, device)
    code = _spread_bits(xyz[:, 0]) | (_spread_bits(xyz[:, 1]) << 1) | (_spread_bits(xyz[:, 2]) << 2)
    xyz = xyz[torch.argsort(code)[:n]]
    out = torch.zeros((n, 4), dtype=torch.int32, device=device)
    out[:, 1:] = xyz.int()
    return out.contiguous()


# ------------------------------------------------------------------------------------------------ the calibration

def _warm(device: Any, seconds: float = 1.0) -> None:
    """The card busy for `seconds` (its clocks up) before the first key is timed."""
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    torch.cuda.synchronize(device)
    began = time.perf_counter()
    while time.perf_counter() - began < seconds:
        for _ in range(8):
            a @ a
        torch.cuda.synchronize(device)
    del a


def _other_rows(n: int, logn: int) -> Optional[int]:
    """A row count of the same bucket as n with the other divisibility by 16 (Triton compiles a kernel apart for an
    int argument divisible by 16: the chosen tiles are compiled for both kinds of level here, not at the artist's
    first decode), or None when the bucket has no such count below n."""
    other = n - 1 if n % 16 == 0 else n - n % 16
    return other if other >= (1 << logn) else None


def _one(n: int, logn: int, ci: int, co: int, seed: int, device: Any, masked: Callable) -> None:
    """One key: a synthetic level of n rows, its neighbour cache (lean_cache.build, as the decoders build it), random
    float16 tables, then the two calls the decoders make: the whole level through flex_gemm's convolution (the
    split-K wrapper: its SPLITK, and the tiles under it), and the masked kernel's own function (chunked.conv_rows:
    the boxes' and slabs' convolutions, SPLITK 1). Each tunes what its tuners do not hold yet. Then both calls once
    on the level's first _other_rows rows (the same key, nothing to tune): the chosen tiles compiled for a row count
    of the other divisibility by 16 as well."""
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d

    from . import lean_cache

    with torch.no_grad():
        coords = level_coords(n, device)
        shape = torch.Size([1, ci, *(coords[:, 1:].max(0).values + 1).tolist()])
        cache = lean_cache.build(coords, shape, (3, 3, 3), (1, 1, 1))
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        feats = torch.randn(n, ci, device=device, dtype=torch.float16, generator=gen)
        weight = torch.randn(co, 3, 3, 3, ci, device=device, dtype=torch.float16, generator=gen)
        weight.mul_(1.0 / math.sqrt(V * ci))
        bias = torch.randn(co, device=device, dtype=torch.float16, generator=gen).mul_(0.1)
        out, _ = sparse_submanifold_conv3d(feats, coords, shape, weight, bias, cache, (1, 1, 1))
        del out
        out = masked(feats, weight.reshape(co, V, ci), bias, cache.neighbor_map, cache.sorted_idx,
                     cache.valid_kernel_callback, cache.valid_kernel_seg_callback)
        del out, cache
        other = _other_rows(n, logn)
        if other is not None:
            part = coords[:other].contiguous()
            cache = lean_cache.build(part, shape, (3, 3, 3), (1, 1, 1))
            rows = feats[:other]
            out, _ = sparse_submanifold_conv3d(rows, part, shape, weight, bias, cache, (1, 1, 1))
            del out
            out = masked(rows, weight.reshape(co, V, ci), bias, cache.neighbor_map, cache.sorted_idx,
                         cache.valid_kernel_callback, cache.valid_kernel_seg_callback)
            del out
        torch.cuda.synchronize(device)


def run(device: Any, say: Callable[[str], None] = print, grid: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        drop: bool = True, pause: Optional[Callable[[], None]] = None) -> dict:
    """Tune every key of `grid` (GRID by default) through flex_gemm's own tuners and save them into its cache file
    (once, at the end, only when every key was done: a run that fails or is interrupted leaves the file as it was and
    the tuners as they were); with `drop` the three tuners' entries of this card are replaced by the calibrated keys
    (a first-use tuning may have been made under load). `pause()` is called before each key (the caller checks the
    card is still quiet). Returns the record's "flex_gemm" section."""
    import importlib

    from flex_gemm import AUTOTUNE_CACHE_PATH
    from flex_gemm.ops import spconv
    from flex_gemm.utils import autotuner

    from . import config as conv_config
    from . import tuning

    grid = dict(GRID if grid is None else grid)
    tuning.install()
    spconv.set_algorithm(conv_config.FLEX_GEMM_ALGO)
    spconv.set_hashmap_ratio(conv_config.FLEX_GEMM_HASHMAP_RATIO)
    masked = getattr(importlib.import_module(f"flex_gemm.kernels.triton.spconv.{tuning.KERNEL}"), tuning.KERNEL)
    tuners = tuning.tuners()
    if len(tuners) != 3:
        raise RuntimeError(f"flex_gemm's convolution tuners are not laid out as expected ({sorted(tuners)})")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    began = time.perf_counter()
    dropped = {name: len(t.cache) for name, t in tuners.items()} if drop else {}
    keys: List[list] = []
    skipped: List[dict] = []
    autosave = getattr(autotuner, "AUTOSAVE_AUTOTUNE_CACHE", None)
    before = {name: dict(t.cache) for name, t in tuners.items()}      # put back should the run not finish
    finished = False
    with tuning.suspended():
        try:
            autotuner.AUTOSAVE_AUTOTUNE_CACHE = False        # one save at the end, not one per key and tuner
            if drop:
                for t in tuners.values():
                    t.cache.clear()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            base_mib = torch.cuda.memory_allocated(device) / 2**20
            _warm(device)
            for (ci, co), (lo, hi) in grid.items():
                fam_began = time.perf_counter()
                done = 0
                for logn in range(lo, hi + 1):
                    n = rows_for(logn, ci, co, sms)
                    if n is None:
                        skipped.append({"key": f"2^{logn} {family(ci, co)}", "why": f"over {MAX_KEY_MIB} MiB"})
                        continue
                    if pause is not None:
                        torch.cuda.synchronize(device)     # the look at the card sees no kernel of this run
                        pause()
                    t0 = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(device)
                    failed = None
                    try:
                        _one(n, logn, ci, co, 7919 * logn + 31 * ci + co, device, masked)
                    except torch.OutOfMemoryError as exc:
                        failed = str(exc).splitlines()[0][:160]
                    torch.cuda.empty_cache()
                    if failed is not None:
                        skipped.append({"key": f"2^{logn} {family(ci, co)}", "why": f"out of memory at {n} rows "
                                        f"({failed})"})
                        say(f"convolutions {family(ci, co)} 2^{logn}: out of memory at {n} rows - skipped (the "
                            f"nearest calibrated size serves it)")
                        continue
                    peak = torch.cuda.max_memory_allocated(device) / 2**20 - base_mib
                    held = tuning.choice(logn, ci, co, V)
                    keys.append([family(ci, co), logn, n, held["splitk"], held["tile"], held["split_tile"],
                                 round(time.perf_counter() - t0, 2), round(peak)])
                    done += 1
                say(f"convolutions {ci}x{co}: {done} sizes (2^{lo}-2^{hi} rows) in "
                    f"{time.perf_counter() - fam_began:.1f} s")
            finished = True
        finally:
            autotuner.AUTOSAVE_AUTOTUNE_CACHE = autosave
            if finished:
                if keys or drop:
                    autotuner.save_autotune_cache()
            else:                                           # the tuners as they were: nothing half done is kept
                for name, t in tuners.items():
                    t.cache.clear()
                    t.cache.update(before[name])
    seconds = time.perf_counter() - began
    peak_key = max((k[7] for k in keys), default=0)
    return {
        "grid": {family(ci, co): [lo, hi] for (ci, co), (lo, hi) in grid.items()},
        "keys": keys,
        "key_fields": ["shape", "log2_rows", "rows", "splitk", "tile", "split_tile", "seconds", "peak_mib"],
        "skipped": skipped, "dropped": dropped,
        "held": {name: len(t.cache) for name, t in tuners.items()},
        "tiles": len(tuners["masked"].configs), "sms": sms,
        "cache_file": str(AUTOTUNE_CACHE_PATH), "peak_key_mib": peak_key,
        "summary": f"{len(keys)} sizes of {len(grid)} shapes tuned in {seconds / 60:.1f} min (the largest "
                   f"{peak_key} MiB of the card){'; ' + str(len(skipped)) + ' skipped' if skipped else ''}",
    }
