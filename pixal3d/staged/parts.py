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
from .runner import FINISH_SAFETY_MIB, NATIVE_RESERVE, PartContext, PropState, glb_reserve
from .types import DecodedMesh, Event, PropFailed

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

# Whether both decoders may be on the card at once in "finish": weights of one, voxels a token decodes to at most,
# and PyTorch's allocated need per thousand decoded voxels.
DECODER_MIB = 905
VOXELS_PER_TOKEN = 311
DECODE_MIB_PER_KVOXEL = 0.80


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
    """A sampler's step callback: progress, the next prop's inputs on their way to the card (not when memory is
    tight), and a cancel check point."""
    def callback(step: int, total: int) -> None:
        ctx.emit(Event("step", part, prop.id, step, total))
        if mode == "normal":
            ctx.store.pump()
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
    cond = ctx.take(prop, "cond_ss")
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
    # The token cap of p2d:711-725: a 1536 prop with too many tokens is made at a lower resolution.
    lr_resolution = 512
    resolution = request.resolution
    while True:
        grid_res = resolution // 16
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
        ], dim=1)
        unique = quant_coords.unique(dim=0)
        if unique.shape[0] < request.max_num_tokens or resolution == 1024:
            break
        resolution -= 128
    store.put_small(prop.id, "coords_hr", {"coords": unique.cpu(),
                                           "res": torch.tensor([resolution], dtype=torch.int32)})


def cond_hr(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    naf = _network(ctx, "naf")
    source, view = _source(ctx, prop)
    stored = ctx.store.get_small(prop.id, "coords_hr")
    coords = stored["coords"].to(ctx.device)
    grid = int(stored["res"][0]) // 16
    z1024 = ctx.take(prop, "dino1024")["z"]
    image = proj_cond.image_tensor(source, 1024, ctx.device)
    shape = _bf16(proj_cond.cond_tokens(proj_cond.SPECS["shape_1024"], z1024, image, naf, view, coords, grid))
    ctx.loader.trim()
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
                          "tex_slat": tex_feats}, dict(store.get_small(prop.id, "notes")))


def decoders_fit(tokens: int, limit_mib: int) -> bool:
    """May both decoders stay on the card while a prop of `tokens` tokens is decoded under a cap of `limit_mib`."""
    need = DECODE_MIB_PER_KVOXEL * VOXELS_PER_TOKEN * tokens / 1000
    return 2 * DECODER_MIB + need <= limit_mib / 1.1


def finish(ctx: PartContext, prop: PropState, mode: str) -> None:
    decode = _decode()
    store, loader, device = ctx.store, ctx.loader, ctx.device
    stored = store.get_small(prop.id, "coords_hr")
    coords = stored["coords"].to(device)
    resolution = int(stored["res"][0])
    shape_feats = store.get_small(prop.id, "shape_slat")["feats"].to(device)
    tex_feats = store.get_small(prop.id, "tex_slat")["feats"].to(device)
    notes = dict(store.get_small(prop.id, "notes"))
    capped = mode != "uncapped"

    # Phase 1, the decoders. Weights go onto the card before anything else of the phase.
    limit = loader.set_cap(NATIVE_RESERVE["finish"], safety_mib=FINISH_SAFETY_MIB) if capped else 0
    together = mode == "normal" and decoders_fit(coords.shape[0], limit)
    ctx.check()
    if together:
        _decoder(ctx, "shape_dec")
        _decoder(ctx, "tex_dec")
    elif loader.get("tex_dec") is not None:         # left by an attempt that failed with both on the card
        loader.drop("tex_dec")
    mesh, subs = decode.decode_shape(_decoder(ctx, "shape_dec"), shape_feats, coords, resolution)
    del shape_feats
    if together:
        loader.trim()
    else:
        loader.drop("shape_dec")
    ctx.emit(Event("step", "finish", prop.id, 1, 3))
    ctx.check()
    tex_coords, attrs = decode.decode_texture(_decoder(ctx, "tex_dec"), tex_feats, coords, subs)
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

    # Phase 2, the caller's hook. Once it has started the prop is finished and saved, cancel or not.
    ctx.check()
    if capped:
        loader.set_cap(glb_reserve(decoded.faces.shape[0]), safety_mib=FINISH_SAFETY_MIB)
    prop.result = dict(ctx.hooks.finish(prop.req, decoded, notes) or {})
    prop.resolution = resolution


TABLE = {"cutout": cutout, "camera": camera, "features": features, "ss": ss, "cond_lr": cond_lr, "lr": lr,
         "upsample": upsample, "cond_hr": cond_hr, "hr": hr, "tex": tex, "finish": finish}
