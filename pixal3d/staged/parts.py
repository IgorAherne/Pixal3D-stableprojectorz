"""The eleven parts of a prop, and what each needs: this is the plan the runner follows (runner.StagedRunner).

A part function is `fn(ctx, prop, mode)`. It reads its inputs from the batch's store, computes with the networks
the loader holds for the part, and writes its results to the store as its LAST actions, after everything that can
raise: an attempt that failed has changed nothing, and the next attempt (mode "tight", then "uncapped": the
runner's ladder) starts from the same inputs. The parts and the lines of the reference pipeline each reproduces
(p2d = pipelines/pixal3d_image_to_3d.py, pmv = pipelines/pixal3d_mv_image_to_3d.py):

    cutout    hooks.prepare: the picture(s) opened, cut out and framed (background removal loaded when first asked for)
    camera    hooks.camera with MoGe: single-view props only
    features  DINOv3 once at 512 px and once at 1024 px; the dense conditioning of the first stage (p2d:193-227)
    ss        the sparse structure (p2d:301-348), through engine.sample_sparse_structure so that a caller's guard
              wrapped around that attribute applies
    cond_lr   conditioning of the 512 shape stage at the prop's tokens (p2d:229-295)
    lr        the 512 shape latent (p2d:350-388)
    upsample  the shape decoder's up-sampling and the token cap (p2d:702-725)
    cond_hr   conditionings of the 1024 shape stage and of the texture stage (p2d:730-736, 762-769)
    hr        the 1024 shape latent (p2d:737-757)
    tex       the texture latent (p2d:510-551); then the checkpoint a restarted batch resumes from
    finish    both decoders and the hole filling (p2d:574-606); the mesh to the CPU; hooks.finish writes the file

The random numbers are the reference's: drawn on the CPU with the reference's shapes, in the reference's order.
`ss` seeds the generator; `ss`, `lr` and `hr` store its state after their draws under their own key, and the next
drawing part starts from the stored state. A part never overwrites the state it started from, so a second attempt
draws the same noise, whatever other props did in between.

The conditioning and decoding functions (pipelines/proj_cond.py, pipelines/decode.py) are imported when a part
first needs them, not with this module: NAF's code asks for the graphics card when it is imported, and building an
engine must not.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch

from ..modules.sparse import SparseTensor
from .runner import PartContext, PropState
from .types import PARTS, DecodedMesh, Event, FinishSteps, PropFailed

# MiB outside PyTorch's allocator while a part runs (cuBLAS / cuDNN handles, flash_attn, natten, flex_gemm work
# areas; for "finish" its decode phase, which adds cumesh's fill_holes). The loader's cap leaves this much free.
NATIVE_RESERVE = {**{part: 200 for part in PARTS if part != "finish"}, "finish": 300}   # starting figures
# Both phases of "finish" leave more of the card alone than the other parts do: the page's viewer may load a
# finished model while the next prop is decoded.
FINISH_SAFETY_MIB = 500

# Parts whose networks run cuDNN convolutions. cuDNN quietly takes another algorithm when the work area of its
# first choice cannot be allocated, and another algorithm rounds differently: under the cap the same seed could
# give another cut-out, camera or conditioning without any error. PyTorch counts the failed allocation, so a
# counter that moved during such a part is treated as an out-of-memory failure of that attempt. Nothing inside
# such a part may swallow an out-of-memory of its own (see `_steps`), or the count would mean two things.
CONV_PARTS = frozenset({"cutout", "camera", "features", "cond_lr", "cond_hr", "ss"})
# Parts that draw from the CPU random generator; each starts from the state its predecessor stored.
DRAWING_PARTS = frozenset({"ss", "lr", "hr", "tex"})
# Parts after which PyTorch's cache is given back at once: the conditioning passes and the up-sampling decoder
# leave large freed blocks that the next prop's part does not reuse.
TRIMMED_PARTS = frozenset({"features", "cond_lr", "upsample", "cond_hr"})

# The networks a group's `enter` loads, in loading order ({kind} = "sv" | "mv"), and those a part loads itself when
# a prop first needs them.
ENTER = {
    "cutout": (), "camera": ("moge",), "features": ("dino",), "ss": ("ss_flow.{kind}", "ss_dec"),
    "cond_lr": ("naf",), "lr": ("lr_flow.{kind}",), "upsample": ("shape_dec",), "cond_hr": ("naf",),
    "hr": ("hr_flow.{kind}",), "tex": ("tex_flow.{kind}",), "finish": (),
}
LAZY = {"cutout": ("rembg",), "finish": ("shape_dec", "tex_dec")}
LIGHT = frozenset({"naf"})              # megabytes: reading its file ahead gains nothing

# Large store items: what a part takes (the runner asks the store for the next prop's ahead of time), what it puts,
# and what nobody reads once the part is done.
WANTS = {"ss": ("cond_ss",), "cond_lr": ("dino512",), "lr": ("cond_lr",), "cond_hr": ("dino1024",),
         "hr": ("cond_hr",), "tex": ("cond_tex",)}
PRODUCES = {"features": ("dino512", "dino1024", "cond_ss"), "cond_lr": ("cond_lr",),
            "cond_hr": ("cond_hr", "cond_tex")}
SPENT = {"ss": ("cond_ss",), "cond_lr": ("dino512",), "lr": ("cond_lr", "rng_ss"),
         "upsample": ("coords_ss", "lr_slat"), "cond_hr": ("dino1024", "prep"), "hr": ("cond_hr", "rng_lr"),
         "tex": ("cond_tex", "rng_hr")}

# What the end of a prop needs, as measured on an RTX 3060 over the captured reference cases and the bulky prop
# (code/logs/profiling/pixal3d-lowmem/staged in Spellcast3D). The decoders: the weights of one; what a decode
# allocates per thousand voxels of the result; and how many voxels a shape of M occupied voxels on the 512 grid
# decodes to at resolution R: about M * (R / 512)^2 (0.97-1.12 of it over the measured props). The flow samplers at
# the 1024 grid: what they allocate at 0 tokens and per thousand tokens, with a quarter on top for the allocator's
# fragmentation under a cap. The finish hook on the CPU: what it holds beside the process's working set, and per
# million voxels of the mesh (its faces, two BVHs, the simplification). These figures decide, in `upsample`, the
# highest resolution a 1536 prop can be made at within the card and the RAM, and in `finish` whether both decoders
# may share the card and whether a decode is hopeless under the cap.
DECODER_MIB = 905
DECODE_MIB_PER_KVOXEL = 0.52
VOXELS_PER_OCCUPIED = 1.15          # decoded voxels per occupied 512-grid voxel scaled to the resolution, with margin
VOXELS_PER_TOKEN = 400              # voxels a token decodes to at most (396 measured on the bulky prop): used only
#                                     where the occupied voxels are not known (a resumed prop)
SAMPLER_BASE_MIB = 2_700
SAMPLER_MIB_PER_KTOKEN = 50
SAMPLER_SLACK = 1.25
HOOK_RAM_BASE_MIB = 3_000
HOOK_RAM_MIB_PER_MVOXEL = 310
RAM_FLOOR_MIB = 1_500               # what must stay available to Windows and the hub beside the hook's own need
# `cond_hr` makes the shape and the texture conditionings together (proj_cond.cond_tokens_pair: NAF's image encoder
# once per view instead of twice, about 0.36 s a view). A single-view prop pays nothing for it on the card; a
# multi-view prop keeps the shape conditioning's running sum in float32 through the texture pass, its widest
# moment, so it goes this way only under a cap of at least this much (a 7.2 GiB card gives about 6,800).
FUSED_COND_MV_CAP_MIB = 7_400


# What a finish hook's GPU step may keep on the card between its steps (types.FinishSteps: state["room_mib"]): a
# cuBVH of the decoded mesh, built for the remesh and needed again for the bake (60-70 bytes a face, 0.6-0.9 GB for
# the meshes of a 1024 prop; cuBVH keeps a copy of its tree in RAM as well). The room is what the card has beside
# what the props still to decode need under the finish cap, or beside what another prop's hook steps need with
# nothing held (HOLD_BESIDE_MIB: the remesh, nvdiffrast, the texture volume and a BVH of their own), whichever is
# more, less a slack; and never more than the RAM allows twice over beside the floor.
HOLD_BESIDE_MIB = 3_000
HOLD_SLACK_MIB = 512
HOLD_RAM_FACTOR = 2


def glb_reserve(faces: int) -> int:
    """MiB of native memory the caller's finish hook needs for a decoded mesh of `faces` faces (to_glb: the
    nvdiffrast context, one BVH at a time, cumesh). A starting figure: too small and the hook spills into
    shared memory, too large and a big mesh is decoded with the cap lifted that would have fitted."""
    return 2300 + 60 * faces // 1_000_000


def voxels_at(occupied: int, resolution: int) -> int:
    """The voxels a shape with `occupied` voxels on the 512 grid decodes to at `resolution`, with margin."""
    return int(VOXELS_PER_OCCUPIED * occupied * (resolution / 512) ** 2)


def decode_mib(voxels: int, decoders: int = 1) -> float:
    """MiB PyTorch allocates to decode a shape of `voxels` voxels with `decoders` decoders on the card."""
    return decoders * DECODER_MIB + DECODE_MIB_PER_KVOXEL * voxels / 1000


def sampler_mib(tokens: int) -> float:
    """MiB a 1024-grid flow sampler needs on the card for a prop of `tokens` tokens, fragmentation included."""
    return SAMPLER_SLACK * (SAMPLER_BASE_MIB + SAMPLER_MIB_PER_KTOKEN * tokens / 1000)


def hook_ram_mib(voxels: int) -> float:
    """MiB of RAM the finish hook takes for a mesh of `voxels` voxels beside what the process already holds."""
    return HOOK_RAM_BASE_MIB + HOOK_RAM_MIB_PER_MVOXEL * voxels / 1_000_000


def card_limits(ctx: PartContext) -> tuple[int, int] | None:
    """(the cap PyTorch will get for the decode phase of `finish`, the cap in force now) in MiB. The first is
    foreseen from the second: the parts before `finish` leave the same networks' room, only the reserves differ.
    None when nothing is capped: the card is not the limit (a CPU run, or the cap was lifted)."""
    now = ctx.loader.stats().get("cap_mib")
    if not now:
        return None
    safety = getattr(ctx.loader, "safety_mib", 300)
    return int(now - ((NATIVE_RESERVE["finish"] + FINISH_SAFETY_MIB) - (NATIVE_RESERVE[ctx.part] + safety))), int(now)


def ram_available_mib() -> int | None:
    """RAM Windows can still hand out, or None where it cannot be asked."""
    try:
        import psutil

        return int(psutil.virtual_memory().available) >> 20
    except Exception:
        return None


def resolution_fits(ctx: PartContext, occupied: int, tokens: int, resolution: int) -> str | None:
    """Why a prop of `tokens` tokens at `resolution` would not fit the memory this PC leaves Pixal3D - "card" or
    "ram" - or None when it fits. The card: the decode of `finish` with one decoder under the cap it will get, and
    the 1024-grid samplers under the cap in force; the RAM: what the finish hook needs beside the process."""
    voxels = voxels_at(occupied, resolution)
    limits = card_limits(ctx)
    if limits is not None:
        finish_limit, cap_now = limits
        if decode_mib(voxels) > finish_limit / 1.1 or sampler_mib(tokens) > cap_now:
            return "card"
    available = ram_available_mib()
    if available is not None and hook_ram_mib(voxels) > available - RAM_FLOOR_MIB:
        return "ram"
    return None


def tail_room(ctx: PartContext, pending: list, held_mib: int) -> int:
    """MiB a finish hook's GPU step may keep on the card when it returns, for a later step of the same prop
    (runner: state["room_mib"]). `pending`: the props of the finish part still to be decoded; `held_mib`: what the
    other props in their steps hold already. 0 when nothing may stay: a CPU run, a prop whose decode size is not
    known, or no room."""
    headroom = ctx.loader.headroom_mib() if hasattr(ctx.loader, "headroom_mib") else None
    if headroom is None:
        return 0
    need = 0.0
    for prop in pending:
        try:
            stored = ctx.store.get_small(prop.id, "coords_hr")
        except Exception:
            return 0
        need = max(need, 1.1 * decode_mib(decode_voxels(stored), decoders=2))
    room = (headroom - (NATIVE_RESERVE["finish"] + FINISH_SAFETY_MIB) - max(need, HOLD_BESIDE_MIB) - held_mib
            - HOLD_SLACK_MIB)
    available = ram_available_mib()
    if available is not None:
        room = min(room, (available - RAM_FLOOR_MIB - HOOK_RAM_BASE_MIB) // HOLD_RAM_FACTOR)
    return max(0, int(room))


def networks(part: str, kind: str) -> tuple[str, ...]:
    """Every network a group of `part` may have on the card."""
    return tuple(name.format(kind=kind) for name in ENTER[part] + LAZY.get(part, ()))


def applies(part: str, prop: PropState) -> bool:
    return part != "camera" or not prop.req.multi


def wants(part: str) -> tuple[str, ...]:
    return WANTS.get(part, ())


def produces(part: str) -> tuple[str, ...]:
    return PRODUCES.get(part, ())


def discards(part: str, prop: PropState) -> tuple[str, ...]:
    keys = SPENT.get(part, ())
    if part == "cond_hr":
        keys += tuple(f"image_{index}" for index in range(prop.views))
    return keys


def enter(ctx: PartContext, part: str, kind: str) -> None:
    """The group's networks onto the card (those the group before left there stay as they are)."""
    for name in ENTER[part]:
        ctx.entering = name.format(kind=kind)
        _network(ctx, ctx.entering)


def leave(ctx: PartContext, part: str, kind: str) -> None:
    """Drop the group's networks, except those the next group needs (ctx.keep)."""
    held = [name for name in networks(part, kind) if name not in ctx.keep and ctx.loader.get(name) is not None]
    if held:
        ctx.loader.drop(*held)


# =====================================================================================================================
# Shared pieces
# =====================================================================================================================

def _network(ctx: PartContext, name: str) -> Any:
    """The network on the card, loaded now if it is not. Never keep what this returns beyond the part function:
    the loader can only free a network nobody else refers to."""
    module = ctx.loader.get(name)
    return module if module is not None else ctx.loader.load(name, ctx.cancelled)


def _decoder(ctx: PartContext, name: str) -> Any:
    """A sparse decoder in its memory-saving form (it gives each level's cache back as soon as the level is done)."""
    decoder = _network(ctx, name)
    decoder.lean = True
    return decoder


def _proj_cond() -> Any:
    from ..pipelines import proj_cond

    return proj_cond


def _decode() -> Any:
    from ..pipelines import decode

    return decode


def _source(ctx: PartContext, prop: PropState) -> tuple[Any, Any]:
    """The prop's picture(s) as the conditioning functions take them, and its camera. A multi-view prop's bundle
    (63 MiB of floats for four views) is built again by every part that needs it and never stored."""
    proj_cond = _proj_cond()
    images = [ctx.store.get_small(prop.id, f"image_{index}") for index in range(prop.views)]
    prep = ctx.store.get_small(prop.id, "prep")
    if prop.req.multi:
        views = ctx.hooks.bundle(prop.req, images, prep["meta"])
        return views, proj_cond.Camera.from_bundle(views, ctx.device)
    return images, proj_cond.Camera.single(prep["camera"], ctx.device)


def _bf16(cond: dict) -> dict:
    """A conditioning as it is stored: the cast the denoisers make of it in every pass, made once."""
    return {name: value.type(torch.bfloat16) for name, value in cond.items()}


def _as_reference_layout(cond: dict, multi: bool) -> dict:
    """The dense conditioning laid out in memory as the reference pipelines hand it to the sparse-structure flow.

    Its first linear layer runs one fused addmm on a contiguous [B, N, D] input, and a matmul plus a separate bias
    add on a permuted view; in bfloat16 the two round differently in every block, so the layout decides the sample.
    Single view reaches the flow as the permuted view ProjGrid.forward ends with, multi view as a contiguous
    tensor. Anything the store gives back is contiguous (it writes its files straight from its own memory), so the
    single-view layout is made again here. It is already right for a tensor that never left the card.
    """
    if multi:
        return cond
    proj = cond["proj"]
    return {**cond, "proj": proj.transpose(1, 2).contiguous().transpose(1, 2)}


def _token_cond(cond: dict, coords: torch.Tensor, negative: bool = True) -> dict:
    """{"cond", "neg_cond"} of a sparse stage: the stored rows on the prop's coords, and zeros of the same form."""
    pair = {"cond": {"global": cond["global"], "proj": SparseTensor(feats=cond["proj"], coords=coords)},
            "neg_cond": None}
    if negative:
        zero = _proj_cond().negative(cond)
        pair["neg_cond"] = {"global": zero["global"], "proj": SparseTensor(feats=zero["proj"], coords=coords)}
    return pair


def _denormalised(slat: SparseTensor, normalization: dict) -> SparseTensor:
    std = torch.tensor(normalization["std"])[None].to(slat.device)
    mean = torch.tensor(normalization["mean"])[None].to(slat.device)
    return slat * std + mean


def _steps(ctx: PartContext, prop: PropState, part: str, mode: str) -> Callable[[int, int], None]:
    """A sampler's step callback: progress, the next prop's inputs on their way to the card (when memory is
    tight they wait for their prop's turn, and what was sent ahead comes back), and a cancel check point.

    Nothing is sent ahead from inside a part whose convolutions the runner watches: the store refuses an
    upload that does not fit by catching the out-of-memory itself, and the runner reads the same counter to
    decide whether cuDNN quietly changed its plan. Only `ss` is both, and it gives up the overlap.
    """
    upload = mode == "normal" and part not in CONV_PARTS

    def callback(step: int, total: int) -> None:
        ctx.emit(Event("step", part, prop.id, step, total))
        ctx.store.pump(upload=upload)
        ctx.check()

    return callback


# =====================================================================================================================
# The parts
# =====================================================================================================================

def cutout(ctx: PartContext, prop: PropState, mode: str) -> None:
    def matte(image: Any) -> Any:
        return _network(ctx, "rembg")(image)

    prepared = ctx.hooks.prepare(prop.req, matte)
    if not prepared.images:
        raise PropFailed("The picture could not be prepared.", "input")
    for index, image in enumerate(prepared.images):
        ctx.store.put_small(prop.id, f"image_{index}", image)
    ctx.store.put_small(prop.id, "prep", {"meta": prepared.meta, "camera": None})
    prop.views = len(prepared.images)


def camera(ctx: PartContext, prop: PropState, mode: str) -> None:
    store = ctx.store
    found = ctx.hooks.camera(prop.req, store.get_small(prop.id, "image_0"), _network(ctx, "moge"))
    store.put_small(prop.id, "prep", {**store.get_small(prop.id, "prep"), "camera": dict(found)})


def features(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    dino = _network(ctx, "dino")
    source, view = _source(ctx, prop)
    z512 = proj_cond.encode(dino, proj_cond.image_tensor(source, 512, ctx.device))
    z1024 = proj_cond.encode(dino, proj_cond.image_tensor(source, 1024, ctx.device))
    cond = _bf16(proj_cond.cond_dense(proj_cond.SPECS["ss"], z512, view))
    ctx.store.put_device(prop.id, "dino512", {"z": z512})
    ctx.store.put_device(prop.id, "dino1024", {"z": z1024})
    ctx.store.put_device(prop.id, "cond_ss", cond)


def draw_sparse_structure(ctx: PartContext, prop: PropState, mode: str, cond: dict, resolution: int,
                          num_samples: int = 1, sampler_params: dict | None = None) -> torch.Tensor:
    """One draw of the sparse structure: coords [N0,4] int32 on the card. What engine.sample_sparse_structure
    runs while part `ss` works on `prop`; a guard around that attribute may call it several times, and every
    call draws new noise from the CPU generator."""
    args = ctx.args(prop)
    flow = _network(ctx, f"ss_flow.{prop.kind}")
    reso = flow.resolution
    noise = torch.randn(num_samples, flow.in_channels, reso, reso, reso).to(ctx.device)
    params = {**args.sampler_params["ss"], **(sampler_params or {})}
    z_s = args.samplers["ss"].sample(
        flow, noise, **cond, **params, verbose=False, keep_trajectory=False,
        step_callback=_steps(ctx, prop, "ss", mode),
    ).samples
    decoded = _network(ctx, "ss_dec")(z_s) > 0
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()


def ss(ctx: PartContext, prop: PropState, mode: str) -> None:
    store, request = ctx.store, prop.req
    cond = _as_reference_layout(ctx.take(prop, "cond_ss"), prop.req.multi)
    pair = {"cond": cond, "neg_cond": _proj_cond().negative(cond)}
    draw = functools.partial(draw_sparse_structure, ctx, prop, mode)
    torch.manual_seed(request.seed)
    with ctx.hooks.prop_scope(request, "ss") as found, ctx.engine.drawing(draw):
        coords = ctx.engine.sample_sparse_structure(pair, 32, 1, request.sampler_params.get("ss", {}))
    rng = torch.get_rng_state()
    if coords.shape[0] == 0:
        raise PropFailed("The first stage found no shape in the picture.")
    store.put_small(prop.id, "coords_ss", {"coords": coords.cpu()})
    store.put_small(prop.id, "rng_ss", {"state": rng})
    store.put_small(prop.id, "notes", {**store.get_small(prop.id, "notes"), **(found or {})})


def cond_lr(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    naf = _network(ctx, "naf")
    source, view = _source(ctx, prop)
    coords = ctx.store.get_small(prop.id, "coords_ss")["coords"].to(ctx.device)
    z512 = ctx.take(prop, "dino512")["z"]
    image = proj_cond.image_tensor(source, 512, ctx.device)
    cond = _bf16(proj_cond.cond_tokens(proj_cond.SPECS["shape_512"], z512, image, naf, view, coords, 32))
    ctx.store.put_device(prop.id, "cond_lr", cond)


def _shape_latent(ctx: PartContext, prop: PropState, mode: str, part: str, coords_key: str, rng_key: str,
                  slat_key: str) -> None:
    """A shape stage (`lr` at the 512 grid, `hr` at the 1024 grid): noise at the prop's tokens from the stored
    generator state, the sampler, the latent denormalised."""
    store, device, args = ctx.store, ctx.device, ctx.args(prop)
    flow = _network(ctx, f"{part}_flow.{prop.kind}")
    coords = store.get_small(prop.id, coords_key)["coords"].to(device)
    pair = _token_cond(ctx.take(prop, f"cond_{part}"), coords)
    params = {**args.sampler_params["shape"], **prop.req.sampler_params.get("shape", {})}
    torch.set_rng_state(store.get_small(prop.id, rng_key)["state"])
    noise = SparseTensor(feats=torch.randn(coords.shape[0], flow.in_channels).to(device), coords=coords)
    slat = args.samplers["shape"].sample(
        flow, noise, **pair, **params, verbose=False, keep_trajectory=False,
        step_callback=_steps(ctx, prop, part, mode),
    ).samples
    slat = _denormalised(slat, args.shape_normalization)
    rng = torch.get_rng_state()
    store.put_small(prop.id, slat_key, {"feats": slat.feats.cpu()})
    store.put_small(prop.id, f"rng_{part}", {"state": rng})


def lr(ctx: PartContext, prop: PropState, mode: str) -> None:
    _shape_latent(ctx, prop, mode, "lr", "coords_ss", "rng_ss", "lr_slat")


def upsample(ctx: PartContext, prop: PropState, mode: str) -> None:
    store, device, request = ctx.store, ctx.device, prop.req
    feats = store.get_small(prop.id, "lr_slat")["feats"].to(device)
    coords = store.get_small(prop.id, "coords_ss")["coords"].to(device)
    hr_coords = _decode().upsample(_decoder(ctx, "shape_dec"), feats, coords)
    occupied = hr_coords.shape[0]

    def tokens_at(resolution: int) -> torch.Tensor:
        grid_res = resolution // 16
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / 512 * (grid_res - 1)).round().int(),
        ], dim=1)
        return quant_coords.unique(dim=0)

    # The token cap of p2d:711-725: a 1536 prop with too many tokens is made at a lower resolution. The memory
    # this PC leaves Pixal3D caps it as well (resolution_fits): a prop that would not fit the card or the RAM at
    # a resolution is made at the next lower one, down to 1024, instead of spilling or failing at the end.
    resolution, unique = request.resolution, None
    unbounded, why = None, None            # the resolution the token cap alone gives, and what memory said against it
    while True:
        unique = tokens_at(resolution)
        under_cap = unique.shape[0] < request.max_num_tokens
        if unbounded is None and (under_cap or resolution == 1024):
            unbounded = resolution
        reason = resolution_fits(ctx, occupied, unique.shape[0], resolution) if under_cap else None
        why = why or reason
        if (under_cap and reason is None) or resolution == 1024:
            break
        resolution -= 128
    bound_by = why if resolution != unbounded else None
    stored = {"coords": unique.cpu(), "res": torch.tensor([resolution], dtype=torch.int32),
              "voxels": torch.tensor([voxels_at(occupied, resolution)], dtype=torch.int64)}
    store.put_small(prop.id, "coords_hr", stored)
    if bound_by is not None:
        notes = store.get_small(prop.id, "notes")
        store.put_small(prop.id, "notes", {**notes, "memory_cap": {
            "by": bound_by, "resolution": resolution, "without": unbounded, "tokens": unique.shape[0]}})


def cond_hr(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    naf = _network(ctx, "naf")
    source, view = _source(ctx, prop)
    stored = ctx.store.get_small(prop.id, "coords_hr")
    coords = stored["coords"].to(ctx.device)
    grid = int(stored["res"][0]) // 16
    z1024 = ctx.take(prop, "dino1024")["z"]
    image = proj_cond.image_tensor(source, 1024, ctx.device)
    limits = card_limits(ctx)
    together = mode == "normal" and (not prop.req.multi or limits is None or limits[1] >= FUSED_COND_MV_CAP_MIB)
    pair_fn = getattr(proj_cond, "cond_tokens_pair", None)      # a conditioning module without it: the two calls
    pair = None
    if together and pair_fn is not None:
        try:
            pair = pair_fn(proj_cond.SPECS["shape_1024"], proj_cond.SPECS["tex_1024"], z1024, image, naf, view,
                           coords, grid, first_dtype=torch.bfloat16)
        except ValueError:          # nothing to share for this NAF or picture: the two calls, as before
            pair = None
    if pair is not None:
        shape, tex = _bf16(pair[0]), _bf16(pair[1])
    else:
        shape = _bf16(proj_cond.cond_tokens(proj_cond.SPECS["shape_1024"], z1024, image, naf, view, coords, grid))
        ctx.loader.trim()       # the texture conditioning is the widest moment of a prop: it starts from a clean cache
        tex = _bf16(proj_cond.cond_tokens(proj_cond.SPECS["tex_1024"], z1024, image, naf, view, coords, grid))
    ctx.store.put_device(prop.id, "cond_hr", shape)
    ctx.store.put_device(prop.id, "cond_tex", tex)


def hr(ctx: PartContext, prop: PropState, mode: str) -> None:
    _shape_latent(ctx, prop, mode, "hr", "coords_hr", "rng_lr", "shape_slat")


def tex(ctx: PartContext, prop: PropState, mode: str) -> None:
    store, device, args, request = ctx.store, ctx.device, ctx.args(prop), prop.req
    flow = _network(ctx, f"tex_flow.{prop.kind}")
    stored = store.get_small(prop.id, "coords_hr")
    shape_feats = store.get_small(prop.id, "shape_slat")["feats"]
    coords = stored["coords"].to(device)
    params = {**args.sampler_params["tex"], **request.sampler_params.get("tex", {})}
    # A guidance strength of 1 runs only the positive pass: the zero conditioning would never be read.
    pair = _token_cond(ctx.take(prop, "cond_tex"), coords, negative=params.get("guidance_strength") != 1)

    std = torch.tensor(args.shape_normalization["std"])[None].to(device)
    mean = torch.tensor(args.shape_normalization["mean"])[None].to(device)
    shape_slat = (SparseTensor(feats=shape_feats.to(device), coords=coords) - mean) / std
    torch.set_rng_state(store.get_small(prop.id, "rng_hr")["state"])
    noise = shape_slat.replace(
        feats=torch.randn(shape_slat.coords.shape[0], flow.in_channels - shape_slat.feats.shape[1]).to(device))
    slat = args.samplers["tex"].sample(
        flow, noise, concat_cond=shape_slat, **pair, **params, verbose=False, keep_trajectory=False,
        step_callback=_steps(ctx, prop, "tex", mode),
    ).samples
    tex_feats = _denormalised(slat, args.tex_normalization).feats.cpu()

    store.put_small(prop.id, "tex_slat", {"feats": tex_feats})
    if request.request_hash:
        store.checkpoint(prop.id, request.request_hash,
                         {"coords": stored["coords"], "res": stored["res"], "shape_slat": shape_feats,
                          "tex_slat": tex_feats}, prop.noted(store.get_small(prop.id, "notes")))


def decoders_fit(voxels: int, limit_mib: int) -> bool:
    """May both decoders stay on the card while a shape of `voxels` voxels is decoded under a cap of `limit_mib`."""
    return decode_mib(voxels, decoders=2) <= limit_mib / 1.1


def decode_voxels(stored: dict) -> int:
    """The voxels the stored coords_hr decode to: as `upsample` foresaw them, or from the token count for a prop
    that was resumed from a checkpoint."""
    voxels = stored.get("voxels")
    return int(voxels[0]) if voxels is not None else VOXELS_PER_TOKEN * int(stored["coords"].shape[0])


def finish(ctx: PartContext, prop: PropState, mode: str) -> "FinishSteps | None":
    decode = _decode()
    store, loader, device = ctx.store, ctx.loader, ctx.device
    stored = store.get_small(prop.id, "coords_hr")
    resolution = int(stored["res"][0])
    notes = prop.noted(store.get_small(prop.id, "notes"))
    capped = mode != "uncapped"
    voxels = decode_voxels(stored)

    # Phase 1, the decoders. A decoder's weights go onto the card before the latents it decodes.
    limit = loader.set_cap(NATIVE_RESERVE["finish"], safety_mib=FINISH_SAFETY_MIB) if capped else 0
    if capped and decode_mib(voxels / VOXELS_PER_OCCUPIED) > limit:
        # Hopeless under the cap, even with every margin taken off: an attempt would run out of memory after a
        # minute and leave nothing. Said at once; the ladder lifts the cap when the caller allows a spill.
        raise PropFailed(f"Decoding this shape ({voxels / 1e6:.1f} M voxels) needs about "
                         f"{decode_mib(voxels / VOXELS_PER_OCCUPIED):,.0f} MiB of video memory; the card has "
                         f"{limit:,} MiB free for it.", "oom")
    together = mode == "normal" and decoders_fit(voxels, limit)
    ctx.check()
    if not together and loader.get("tex_dec") is not None:      # left by an attempt that failed with both loaded
        loader.drop("tex_dec")
    for name in LAZY["finish"] if together else LAZY["finish"][:1]:
        _decoder(ctx, name)
    coords = stored["coords"].to(device)
    shape_feats = store.get_small(prop.id, "shape_slat")["feats"].to(device)
    mesh, subs = decode.decode_shape(loader.get("shape_dec"), shape_feats, coords, resolution)
    del shape_feats
    if together:
        loader.trim()
    else:
        loader.drop("shape_dec")
    ctx.emit(Event("step", "finish", prop.id, 1, 3))
    ctx.check()
    _decoder(ctx, "tex_dec")
    tex_feats = store.get_small(prop.id, "tex_slat")["feats"].to(device)
    tex_coords, attrs = decode.decode_texture(loader.get("tex_dec"), tex_feats, coords, subs)
    del tex_feats, subs, coords
    loader.drop(*(name for name in LAZY["finish"] if loader.get(name) is not None))
    ctx.emit(Event("step", "finish", prop.id, 2, 3))
    decode.fill_holes(mesh)
    ctx.emit(Event("step", "finish", prop.id, 3, 3))

    # Nothing of the prop stays on the card: the hook's own work needs the room, and uploads what it needs.
    decoded = DecodedMesh(vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(), coords=tex_coords.cpu(),
                          attrs=attrs.cpu(), resolution=resolution, attr_layout=ctx.args(prop).pbr_attr_layout)
    del mesh, tex_coords, attrs
    loader.trim()

    # Phase 2, the caller's hook. Once it has started the prop is finished and saved, cancel or not. The hook may
    # hand the rest back as steps (types.FinishSteps): the runner then runs them around the next props' decodes,
    # the CPU ones on its worker thread.
    ctx.check()
    reserve = glb_reserve(decoded.faces.shape[0])
    if capped:
        loader.set_cap(reserve, safety_mib=FINISH_SAFETY_MIB)
    prop.resolution = resolution
    outcome = ctx.hooks.finish(prop.req, decoded, notes)
    if isinstance(outcome, FinishSteps):
        outcome.native_mib = outcome.native_mib or reserve
        outcome.safety_mib = FINISH_SAFETY_MIB if outcome.safety_mib is None else outcome.safety_mib
        return outcome
    prop.result = dict(outcome or {})
    return None


TABLE = {"cutout": cutout, "camera": camera, "features": features, "ss": ss, "cond_lr": cond_lr, "lr": lr,
         "upsample": upsample, "cond_hr": cond_hr, "hr": hr, "tex": tex, "finish": finish}
