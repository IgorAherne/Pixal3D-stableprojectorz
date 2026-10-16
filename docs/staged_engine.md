# The staged engine

`pixal3d/staged/` runs the Pixal3D cascade in little memory. The reference pipelines
(`Pixal3DImageTo3DPipeline.run`, `Pixal3DMVImageTo3DPipeline.run_mv`) keep every network in RAM and move one after
the other to the card for each prop. The staged engine makes a whole batch of props **part by part**: every prop
goes through one part before any prop starts the next, and a part's networks are read from disk straight onto the
card when the part starts and dropped when it ends. No network rests in RAM (what a prop carries from part to
part does, within a budget, and spills to the work folder beyond it), and PyTorch's reserve on the card is capped
so that it recycles its cache instead of spilling into shared memory.

The results are those of the reference pipelines for the same picture, seed and settings: same random numbers in
the same order, same arithmetic. `run()` and `run_mv()` stay in the fork as that reference.

## Using it

```python
from pathlib import Path
from pixal3d.staged import StagedEngine, EngineOptions, PropRequest
from pixal3d.staged.builders import default_builders, default_paths

options = EngineOptions(pipeline_dir=snapshot,                       # pipeline.json, pipeline_mv.json, ckpts/
                        builders=default_builders(default_paths(models_root)))
engine = StagedEngine(options, hooks)                                # reads two JSON files; loads nothing
                                                                     # (reads background removal's file ahead)

requests = [PropRequest(id="chair", seed=11, resolution=1024, multi=False,
                        sampler_params={"ss": {...}, "shape": {...}, "tex": {...}})]
results = engine.run(requests, Path(work_dir), on_event=print, cancelled=lambda: False)
# {"chair": {"state": "done", "result": {..., "slow": [], "seconds": 187.2, "resolution": 1024, "resumed": False}}}

engine.release()        # before anything else needs the card (the reference pipeline, another tool)
engine.close()          # at exit: also stops the helper threads
```

- `run` blocks and must be called on the one thread that does all CUDA work of the process. Of the failures a
  batch can meet it raises only `BatchBroken` (the CUDA context is unusable; restart the process): anything else
  is laid on the props that had not finished, so the ones that had keep their results. Being called wrongly still
  raises (`ValueError` for two props with one id, `RuntimeError` for a second batch at the same time). A prop that
  fails does not stop the others:
  its entry is `{"state": "failed", "error": {"kind", "detail"}}` with kind `input`, `mv_missing`, `oom`, `ram`,
  `disk`, `missing_file` or `error`.
- `cancelled()` is asked before every prop of every part, between the tensors of a network being loaded, at every
  sampler step and between the steps of the last part. Unfinished props come back as `{"state": "cancelled"}`. A
  prop whose file is being written is finished first.
- `work_dir` belongs to the batch. The engine empties it at the start, except for the checkpoints `resume` is
  about to read, and removes each prop's folder as that prop ends.
- `wave_size=K` makes K props at a time go through all parts (the first results appear sooner, a crash loses
  less). `None` / `0`: all props of the batch together.
- `resume=True` on a second call with the same `work_dir` (after the process died): props whose sampling was
  finished are only decoded and saved. A prop is recognised by its `id` and `request_hash`, which the caller makes
  from everything that defines the result. Props without a `request_hash` are never resumed.

### Hooks

The engine never touches a prop's files. The caller passes an object with five methods (`types.Hooks`), all called
on the GPU thread; each may raise `PropFailed(detail, kind)` to fail its prop:

| hook | when | gives |
|---|---|---|
| `prepare(req, matte)` | part `cutout` | `Prepared(images, meta)`: single view `[RGB picture]`, cut out and framed; multi-view the framed RGBA views, front first. `matte(rgb picture)` is background removal on the card; its network is loaded the first time it is called |
| `camera(req, image, moge)` | part `camera`, single-view props | `{"camera_angle_x", "distance", "mesh_scale"}` as Python floats |
| `bundle(req, images, meta)` | part `features` (again in `cond_lr` / `cond_hr` only where the RAM had no room to keep it), multi-view props | the views dict of `run_mv` (`images` at 512 and 1024, `camera_angle_x`, `camera_distance`, `transform_matrix`, `mesh_scale`) |
| `prop_scope(req, "ss")` | around the sparse-structure stage | a context manager yielding a dict the hook has filled when the block is left; it becomes part of the prop's notes |
| `finish(req, mesh, notes)` | part `finish` | writes the prop's file from the `DecodedMesh` (four CPU tensors) and returns what the result should carry - or a `FinishSteps` (below) |

`engine.sample_sparse_structure(cond, resolution, num_samples, sampler_params)` is the one attribute a caller may
wrap, exactly as the pipeline's method of that name: part `ss` calls it through the engine, positionally. A wrapper
that calls it again draws new noise from the prop's generator, and the state stored for the next stage is the one
after all draws.

#### Finishing in steps

Most of what a finish hook does after the decoders (the UV unwrap, the texture clean-up, writing the file) is CPU
work that leaves the card idle. Instead of a result, `finish` may return `types.FinishSteps(steps, state, result)`:
`steps` is a list of `("gpu" | "cpu", fn)`, each `fn(state)` reading and writing the dict `state`; `result(state)`
gives the prop's result once every step ran. The runner runs the `"gpu"` steps on the GPU thread, between the
other props' turns, under the memory cap (`native_mib`, by default the engine's own figure for a mesh of this
size, and `safety_mib`); the `"cpu"` steps run on `runner.WORKERS` (2) worker threads meanwhile, so the next prop is
decoded while this one is unwrapped, and one prop's export does not wait behind another's BVH build. Rules: a GPU step may run twice after an out-of-memory (it must compute from its inputs and
may overwrite its outputs) and leaves nothing on the card when it returns; a CPU step touches neither the card
nor the engine (no events, no store). An exception in a step fails the prop as the hook's own would; a cancel does
not stop a prop that has reached its steps. At most `runner.IN_FLIGHT` (3) props are in their steps at once, which
bounds the RAM their meshes and textures hold; a prop in its last step (the hub's export, which holds little beside
the file it writes) no longer counts. The hook's steps are reported as `step` events `4..3+n` of `3+n`;
`prop_done` comes when the last step is over, so a prop's turn on the card and its `prop_done` are no longer back
to back. Two more rules of the runner keep the RAM and the tail down (round G): a tail that has run no step yet and
begins with a CPU step is handed to the worker after the GPU steps of the other tails at that boundary, not before
them, while props are still to be decoded (what its first step makes - the hub's cuBVH tree, a gigabyte for a 1536
prop - is needed one texture later, and the GPU steps that run now let go of an older prop's: two trees in RAM at
that moment, not three; the last prop's tail is not deferred); and the props of a paired group go through the
texture and the finish biggest shape first (`parts.pair_order`, by the voxels `upsample` foresaw), so that the last
prop's steps, which nothing hides, are the smallest prop's. A prop's numbers do not depend on its place: every
part draws from the generator state its own prop stored.

One thing a step may leave on the card: what the engine has room for. Before every step the runner writes
`state["room_mib"]`, the MiB the step may keep for a later step of the same prop (`parts.tail_room`: what the card
has beside PyTorch's reserve, less the finish part's reserves, less what the props still to be decoded need under
the cap at most - both decoders with the margin of `decoders_fit` - or what another prop's hook steps need with
nothing held (`parts.HOLD_BESIDE_MIB`), whichever is more, less what the other props in their steps hold already
and a slack; and no more than the RAM allows twice over beside the floor, for things that keep a copy in RAM). A
step that keeps something writes the MiB it holds into `state["card_mib"]` - a CPU step as soon as it starts
taking it, and until a step of the prop has held something the runner counts `FinishSteps.holds_mib`, the most a
step takes, from the offer of a CPU step until that step is over, since the step may not have written the key yet;
once a step has held, the key alone counts (a later CPU step that says nothing holds nothing); the step that uses
the hold up removes the key. A CPU step may touch the card for such a hold and for nothing else. Every `step`
event says what room the step was offered and what the prop holds now (`detail` room_mib, card_mib). `FinishSteps.release(state)` lets go of the hold at once: the runner calls
it on every tail when a decode runs out of memory, before the retry. The runner hands every tail's CPU step to
the worker before it runs any GPU step, so the worker never waits behind another prop's GPU step, and up to
`runner.IN_FLIGHT` (3) props are in their steps at once. Spellcast3D's GLB steps hold the cuBVH of the decoded
mesh this way: its CPU step builds it on the worker thread (cuBVH builds its tree on the CPU for 5-7 s; the
Spellcast3D build of cumesh lets go of Python's lock meanwhile) while the next prop is decoded, the remesh and
the bake both use it (60-70 bytes a face, 0.6-0.9 GB on the card and in RAM for the meshes of a 1024 prop), and
where the card has no room the GPU thread builds it twice as before.

### Events

`on_event(Event)` is called on the GPU thread and must not block.

| kind | meaning |
|---|---|
| `part_start` / `part_end` | a part begins / is over for the wave; `detail["props"]` lists the props in run order, `detail["wave"]` / `["waves"]` the wave |
| `prop_start` | a prop goes onto the card for this part; `detail["position"]` of `detail["of"]` |
| `step` | `step` of `total`: a sampler step, or one of the three steps of `finish` (shape decoded, texture decoded, holes filled), then the hook's own steps when it finishes in steps |
| `prop_rest` | the prop is done with this part and waits for the next |
| `prop_done` | the prop's file is written; `detail["result"]`, `detail["seconds"]` (the GPU thread's seconds on the prop; the worker's are not counted) |
| `prop_failed` | `detail["kind"]`, `detail["detail"]` |
| `prop_spilled` | the memory cap was lifted for this prop in this part (see the ladder) |
| `prop_resumed` | the prop's latents came from a checkpoint |
| `net_load`, `net_ready`, `net_drop`, `store_warning` | the loader's and the store's own events, passed on with their fields in `detail` |

## The parts

| part | networks on the card | reads | writes |
|---|---|---|---|
| `cutout` | background removal, when first asked for | - | `image_<i>`, `prep` |
| `camera` (single view) | MoGe | `image_0` | `prep` (the camera) |
| `features` | DINOv3 | pictures, `prep` | `dino512`, `dino1024`, `cond_ss` |
| `ss` | sparse-structure flow + its decoder | `cond_ss` | `coords_ss`, `rng_ss`, `notes` |
| `cond_lr` | NAF | `dino512`, pictures, `prep`, `coords_ss` | `cond_lr` |
| `lr` | shape flow 512 | `cond_lr`, `coords_ss`, `rng_ss` | `lr_slat`, `rng_lr` |
| `upsample` | shape decoder | `lr_slat`, `coords_ss` | `coords_hr` (with the resolution used) |
| `cond_hr` | NAF | `dino1024`, pictures, `prep`, `coords_hr` | `cond_hr`, `cond_tex` (both at once, `proj_cond.cond_tokens_pair`: NAF's image encoder once per view, every view added into the two sums in place, a head's values through the attention 128 channels at a time at 1024 px; the two calls are the ladder's tight rung) |
| `hr` | shape flow 1024 | `cond_hr`, `coords_hr`, `rng_lr` | `shape_slat`, `rng_hr` |
| `tex` | texture flow | `cond_tex`, `coords_hr`, `shape_slat`, `rng_hr` | `tex_slat`, the checkpoint |
| `finish` | shape decoder, texture decoder | `coords_hr`, `shape_slat`, `tex_slat`, `notes` | the prop's file, through the hook |

The two last parts go prop by prop (`EngineOptions.paired_finish`, on by default): a prop is decoded and handed
to its finish as soon as its texture is sampled, instead of after every prop's texture. The CPU work of its finish
steps (the BVH, the UV unwrap, the file) then runs on the worker threads while the next prop's texture is sampled -
a minute and more at 1536 - rather than only while the next prop is decoded, and only the last prop's steps are left
with nothing to hide under. What the store sent ahead for the next prop's texture goes back to its slot for the
decode. Where the card has room (`parts.pair_keep`, below), the texture network stays on the card through a prop's
decode and its finish steps when another prop of the group follows, and both decoders stay from that prop's finish
to the next one's when there is room for them as well: neither is read again for the next prop. Elsewhere - a card
of 8 GB, the last prop of a group - the texture network leaves the card before the decode (the decode needs the
room, and the cap `finish` sets counts what is allocated as room) and is read again for the next prop, from the
file cache in most cases (its file is read ahead again meanwhile), and the decoders are loaded for every prop, as
before. Whatever is kept for the next prop leaves the card at any failure, before the retry: the ladder's next rung
finds the card as it would have without it. Props resumed from a checkpoint, which run `finish` alone, go
first in their kind. The events are those of the two parts with overlapping brackets: `tex` opens before its
first prop, `finish` before its first prop, `tex` closes after its last prop, `finish` after the last prop's steps
are over. A process that dies in a prop's finish leaves that prop's checkpoint alone: the props after it have not
sampled their texture yet. With `paired_finish=False` every prop's texture is sampled, then every prop is finished,
as the other parts go.

The flow networks exist once per kind (`<name>.sv` from `pipeline.json`, `<name>.mv` from `pipeline_mv.json`);
inside every part the multi-view props go first, then the networks are exchanged and the single-view props follow.
The decoders, DINOv3, NAF and background removal serve both kinds.

The flows' transformer blocks run their elementwise chains (LayerNorm32 and the adaLN modulation, the gated
residuals, the q / k RMS norm with the rotary embedding) as fused Triton kernels (`pixal3d/modules/fused.py`), on
by default: `PIXAL3D_FUSED=off` in the environment gives the eager PyTorch ops, bit for bit the numbers of before.
The kernels reproduce every bf16 rounding of the eager chain; only its reductions are summed in another order, and
the samplers amplify that - as they amplify any change of card or cuBLAS - into a slightly different sample of the
same quality. A detail-flow pass is 11-14 % faster, a sparse-structure pass 21 %, a batch 8-10 %; the memory peak
of a pass is unchanged. Before the first fused pass of a process the kernels are tried against the eager ops on
small tables; a card or a Triton that cannot run them turns the switch off with a `[FUSED]` warning and the flows
run eagerly.

The full self-attention of one prop's tokens (the sparse flows: lr, hr, tex; the dense sparse-structure flow's 4096
tokens) runs SageAttention (`pixal3d/modules/sage.py`: int8 Q.K^T, fp16 P.V with fp32 accumulation, the wheel in
`whl/`) on sm_80+ cards, on by default after a first-use self-check: `PIXAL3D_SAGE=off` gives flash_attn, bit for
bit the numbers of before. flash_attn and cuBLAS already run at the card's bf16 tensor peak, so this cheaper number
format is what moves the samplers: an hr or tex pass at 1536 is 26 % faster, a batch about 20 %, the memory peak of
a pass unchanged. Its error at the attention output is about 4x the bf16 rounding of flash_attn's output, which the
samplers amplify into a different sample of the same quality (Chamfer distance 0.17-0.24 % of the bounding-box
diagonal on the five test props, the class of the fused kernels' own change). The cross-attention (5 context tokens)
and every other call keep their backend.

Two savings around the attention change no bit. A guided sampling runs the flow once on the prop's conditioning
and once on an all-zero conditioning per step inside the guidance interval (hr and lr: 9 of 21 passes, ss: 10 of
22). On the zero one, every block's cross-attention branch (norm2, to_q, the q/k norms, the attention over the five
zero context tokens, to_out, proj_linear) answers the same row for every token, and that row depends on the weights
alone: to_kv of zeros is its bias, so the five keys and values are identical, the softmax is exactly uniform and
the attention returns the value row whatever the query; to_out of identical rows gives identical rows; proj_linear
of zeros is its bias. The engine marks the zero conditioning (`"zero": True` in neg_cond, `parts._token_cond` and
`proj_cond.negative`); the block runs the branch in full on the first zero pass of a token count and keeps its first
row (`SparseProjectAttention._zero_rows`, keyed by the token count, since cuBLAS may pick another kernel for
another count), and the next zero passes of that count add the kept row instead: a zero hr pass at 1536 4.72 ->
4.24 s, a whole hr sampling 85.5 -> 82.2 s, ss 11.0 -> 10.4, lr 5.4 -> 5.1 (about 21 s of a five-prop batch); the
golden engine check and the fused + SageAttention samplings are bit-equal with and without it
(`code/devtools/pixal3d-lowmem/round-i/zero/`). And SageAttention's scaling of v by 2^-8 into the kernel and of
the answer by 2^8 after it is folded into the weights when the switch is on (`sage.fold`, called by the engine's
`_prepare_flow` once a flow is loaded: to_qkv's v rows and bias times 2^-8, to_out's weight times 2^8; powers of
two scale every product and sum exactly, so flash_attn, SDPA and the int8 layers give the same bits too): the
kernel is called without its two elementwise passes (`sage.attention(..., scaled=True)`).

DINOv3 (part `features`) runs under float16 autocast on cards from sm_75 up (`proj_cond.dino_fp16`, switch
`PIXAL3D_DINO_FP16` auto/on/off; a GTX 10xx keeps float32, where fp16 is slow): the features part 3.6x faster
(18.7 -> 9.1 s of a five-prop batch at 1536), the tokens within 7e-4 relative L2 of float32, a different sample of
the same quality (Chamfer 0.16-0.24 % of the bounding-box diagonal on the five test props, the class of the fused
kernels' and SageAttention's changes); off = the float32 tokens, bit for bit the numbers of before. The background
removal (part `cutout`, BiRefNet) runs under float16 autocast on the same cards (`builders.matte_fp16`, switch
`PIXAL3D_MATTE_FP16` auto/on/off): 0.58 -> 0.35 s a view, the 8-bit alpha one or two levels off on a quarter of the
pixels - a cut-out of the same quality, not the same bytes.

A part function is `fn(ctx, prop, mode)` in `parts.py`. It reads from the batch's store, computes, and writes its
results **as its last actions**, so an attempt that failed has changed nothing.

### What is kept between the parts

Everything a prop needs later lies in the batch's `PropStore`, under the prop's id:

| key | content |
|---|---|
| `request`, `notes` | the request's numbers; what the hooks noted (`ss_retries`, `ss_blocked`, ...) and what the engine noted: `memory_cap` when the memory, not the token cap, chose the resolution (`{"by": "card" \| "ram", "resolution", "without", "tokens"}`). The finish hook and the checkpoint get the notes with `slow`: the parts that ran with the cap lifted |
| `image_<i>`, `prep` | the prepared pictures (PIL); `{"meta": ..., "camera": {...}}` |
| `dino512`, `dino1024` | `z` [V,1029,1024] / [V,4101,1024] float32: DINOv3's tokens, computed once per input size |
| `cond_ss` | `global` [1,5,1024], `proj` [1,4096,1024] bfloat16 |
| `cond_lr`, `cond_hr`, `cond_tex` | `global` [1,5,1024], `proj` [tokens,2048] bfloat16 |
| `coords_ss`, `coords_hr` | `coords` [tokens,4] int32 (batch, x, y, z), never re-ordered; `coords_hr` also `res` int32 [1] (the resolution used) and `voxels` int64 [1] (what the decode is foreseen to produce; see the memory rules) |
| `lr_slat`, `shape_slat`, `tex_slat` | `feats` [tokens,32] float32, denormalised |
| `rng_ss`, `rng_lr`, `rng_hr` | `state`: `torch.get_rng_state()` after the part's draws |
| checkpoint | `coords`, `res`, `shape_slat`, `tex_slat` and the notes, on disk: what a restarted batch resumes from |

Conditionings are stored in bfloat16, cast on the card with `.type(torch.bfloat16)`: the cast the denoisers make
of them in every pass. Items are thrown away as soon as their last reader is done (`parts.SPENT`). A multi-view
prop's view bundle (`hooks.bundle`: 47 MiB for three views) is built by `features` and kept as the small item
`bundle` for `cond_lr` and `cond_hr` (spent after `cond_hr`), where the RAM keeps `parts.BUNDLE_RAM_MIB` (3,000)
available beside it; where it does not, each of the three parts builds it again, from the same pictures.

### Random numbers

The reference draws from PyTorch's global CPU generator: once per draw of the sparse structure, then once per
latent. In a batch the props' stages interleave, so the generator state travels with each prop: `ss` seeds it
(`torch.manual_seed(seed)`), and `ss`, `lr`, `hr` store its state after their draws under **their own** key; the
next drawing part restores the stored state first. A part never overwrites the state it started from, so a second
attempt of a part draws the same noise. Noise is always drawn on the CPU with the reference's shapes.

### Memory: the cap and the ladder

Before every prop of every part the runner asks the loader to cap PyTorch's reserve to what is free on the card
now, minus a reserve for native code (`runner.NATIVE_RESERVE`; `finish` sets its own cap twice, once for the
decoders and once for the hook, which needs most of the card for native code). Under the cap an allocation that
does not fit raises instead of spilling. The runner then goes down a ladder, for that prop in that part only:

1. `normal`;
2. `tight`: after the cache was emptied; samplers stop bringing the next prop's inputs to the card ahead of time,
   `finish` holds one decoder at a time;
3. only with `EngineOptions.spill=True`, `uncapped`: the cap is lifted, PyTorch may spill into shared memory as the
   reference does: slow, but done. The part is listed in the result's `slow` and a `prop_spilled` event is sent;
4. then the prop fails with kind `oom`.

By default (`spill=False`) nothing a batch does goes beyond the card's own memory: a prop that does not fit after
the tight attempt fails with `oom` instead of running slowly in shared memory. Only out-of-memory failures are
tried again (`runner.classify`). Parts that run cuDNN convolutions are also treated as failed when PyTorch counted
a failed allocation without raising: cuDNN would have taken another algorithm, which rounds differently.

Three rules keep that failure rare (`parts.resolution_fits`, `parts.finish_mib`, `parts.chunk_for`, with the
figures at the top of `parts.py`, measured on an RTX 3060):

- **A big shape is decoded in boxes.** A shape whose whole-table decode would take more than
  `parts.CHUNK_ABOVE_MIB` (3,000 MiB: every 1536 prop of the test set, no 1024 one) is decoded a box of
  `parts.CHUNK_VOXELS` (2 M) fine voxels at a time (`models/sc_vaes/chunked.py`, `decoder.chunk_voxels`): the
  same vertices and faces, bit for bit, in a peak that no longer grows with the prop - about 0.13 MiB per thousand
  voxels plus 600 MiB for the boxes, 3.4 GB where the whole-table decode of a 16.45 M-voxel shape took 9.2 GB -
  for 4-9 % of the decode's time. What then bounded a big prop on the card was the closing of its holes (cumesh's
  native tables: `parts.fill_mib`), the widest moment of `finish` (`parts.finish_mib`) - about 67 bytes a face
  since round G (cumesh 0.0.1.post2 frees its edge sort's buffers and sizes its tables exactly, `Mesh.fill_holes`
  takes no clones of the mesh; 160 before), so that the decode in boxes is the widest moment again.
- **The resolution of a 1536 prop follows the memory as well as the token cap.** `upsample` walks the resolutions
  down as the reference does (1536, 1408, ..., 1024) and stops at the first whose token count is under the cap
  AND whose finish (`finish_mib`: the decode, in boxes for a big shape, or the closing of the holes) fits the cap
  `finish` will get, whose samplers fit the cap in force, and whose finish hook leaves the RAM a floor; at 1024 it
  stops whatever the figures say. When the memory, not the token cap, chose, the notes carry `memory_cap`. With
  the decode in boxes, every prop of the test set is made at 1536 on a card that leaves Pixal3D 7.2 GiB; what can
  still lower one there is the closing of the holes of a far bulkier shape, or the RAM.
- **A finish that cannot fit under the cap is not attempted.** `finish` fails at once with `oom` (an attempt would
  run out of memory after a minute and leave nothing); with `spill=True` the ladder then runs it with the cap
  lifted, as before. What the pair keeps on the card beside the decode (the texture network) is counted out of
  the cap first.

Two rules use the room a big card leaves, and change nothing where it is short:

- **The paired texture and finish keep their networks on the card where it has room** (`parts.pair_keep`, asked
  by `runner._set_aside` after a prop's texture, when another prop of the group follows). The texture network
  stays through the prop's finish when, with it on the card, the decode with both decoders fits the cap the finish
  will set (with `decoders_fit`'s margin), the closing of the holes fits beside it, and the finish steps' GPU steps
  do within what their cap leaves the card: their native reserve (`glb_reserve` of the faces the voxels foresee -
  `FACES_PER_VOXEL` 2.2 - or the larger one of a prop in its steps) and what they reserve through PyTorch
  (`HOOK_TORCH_MIB` 1,500: Spellcast3D's geometry and bake at the default 2048 texture, 1.0-1.4 GB measured; a
  4096 texture's bake takes 2.5 GB and may then run out of memory once, to run again with what is kept gone);
  both decoders stay as well when the holes and the steps fit beside them too and the next prop's texture sampler
  (`sampler_mib`) fits beside them under the cap in force. The sizes are the loaded networks' own
  (`loader.size_mib`: 2,708 MiB for a texture network, 914 for a decoder). What is kept is counted out of the room a
  finish step is offered (`tail_room`). On the full RTX 3060 every prop of the test set keeps both; under the caps
  of a card that leaves Pixal3D 7.2 GiB nothing is kept, as before (a 1536 prop's decode with both decoders beside
  the texture network needs 7,181 MiB of a 5,749 budget; a 1024 prop's 6,138 of 5,516 under a 7,168 MiB cap).
  Measured on golden props at 1024 on the full card, the pair's loads went from 14 (8.9 s) to 5 (1.4 s) and its
  drops from 15 to 6; a prop of the same kind turns round about 1.1 s sooner and decodes 0.4-0.9 s sooner; under a
  7,168 MiB cap the loads and drops are those of before, and the peaks too (4,116 MiB reserved).
- **The decoders give their cache back level by level only where the decode needs it** (`parts.lean_needed`):
  `decoder.lean` (an `empty_cache` between the levels, the results moved through RAM once to leave the card
  unfragmented) costs about 0.2 s a decoder; without it a decode reserves up to 2.1 times what it allocates on
  whole tables and 1.3 times in boxes (`UNLEAN_WHOLE`, `UNLEAN_BOXES`, measured with no cap on sv1024 and sv1536),
  so the lean form is used where that would not fit the cap less what is kept beside it, with `decoders_fit`'s
  margin, and on the ladder's tight rung. The values are the same either way (`check_decode.py --no-trim`: 12 of 12
  comparisons on both cases). On the full RTX 3060 no prop of the test set needs it; on a card that leaves
  Pixal3D 7.2 GiB the bulkiest 1536 props (12.3 M voxels foreseen) and any shape on whole tables keep it, while the
  golden 1024 props and sv1536 (8.7 M), decoded in boxes, do without. The up-sampling keeps it.

Before the decode in boxes, a bulky prop (41.6k tokens at 1408 by the token cap alone, 16.45 M voxels) was thus
made at 1024 on a card that leaves Pixal3D 7.2 GiB, inside it, in 299 s - instead of at 1408 in 801 s with 2.6 GB
of its decode spilled into shared memory. By the rules now its decode fits at 1408 and the closing of its holes
(5.9 GB there) did not: it was made at 1280 (foreseen by the figures, not measured); with the leaner hole filling
(2.5 GB there by the figures) its decode is the widest moment and it is made at 1408 - its token cap.

Nothing is retried, closed or freed while an exception is being handled: its traceback holds the failed attempt's
tensors. The runner reduces a failure to strings, leaves the `except` block and only then acts.

## Reading the networks: the copies in the models' dtypes

A part's network is read from disk onto the card when the part starts (`loader.py`): built without weights, its
file read in file order, each tensor uploaded and cast on the card to the dtype the model keeps it in. Onto the
card, `loader.READERS` (3) threads read the file at once, `CHUNK_BYTES` (16 MiB) at a time, into slots of
page-locked RAM - the 64 MiB read buffer and a pool of `POOL_BYTES` (64 MiB) beside it, made at the first load and
given back by `close()` - and the copies to the card are queued on the stream while the next chunks are read; a
slot is read into again once an event behind its copies has completed. Every tensor lands in a block of the same
size, taken in the same order, with the same bytes as the one-buffer way gives it (a tensor larger than the buffer
assembled in a block of its own, a smaller one that is cast passing through the one block taken for the casts), and
`cancelled()` is still asked once per tensor. The file cache gives this PC about 11 GB/s whatever the number of
readers; a flow's 2.8 GB load from the cache takes 0.40-0.47 s instead of 0.71-0.82 s, a third of it the model's
construction on the meta device. Onto the CPU (tests) the file goes through the one buffer, tensor by tensor. The
first network a batch loads (background removal) is read into the system's file cache by the read-ahead thread
when the engine is made, and again after a batch's last network has been loaded, for the next batch (that read is
left to go on when the batch ends). The four flow checkpoints are stored in float32 (5.4-5.6 GB each), but the
models keep 480 of
their 700 tensors, 98.7 % of the bytes, in bfloat16: half of what is read is cast away.

`weights.py` writes, once, a copy of such a checkpoint beside it, `<ckpt>.model-dtypes.safetensors`, with every
tensor the model takes from the file already in the model's dtype (2.72 GB for the sparse-structure flow, 2.81 GB
for each other flow: 11.14 GB a set of four instead of 22.0 GB):

```
python -m pixal3d.staged.weights <folder of pipeline.json>      # every checkpoint of pipeline.json, and of
                                                                 # pipeline_mv.json when its models are there
```

```python
from pixal3d.staged import weights
ok = weights.make_copies(snapshot, say=print)    # the same in-process; False when a checkpoint got no copy
weights.make_copy(ckpt)                          # one checkpoint, its path without extension
```

- **What a copy holds.** The tensors the loader takes from the original (the model's state, nothing else), in the
  original's order - the loader puts them on the card in file order, so where each block lands does not change -
  each in the model's dtype, cast on the CPU with `tensor.to(dtype)`: round to nearest even, the cast the loader
  makes on the card. The original is read through a 64 MiB buffer and each tensor (or piece of one) is cast and
  written before the next is read; the copy is written as `<copy>.tmp`, flushed to the disk and renamed. The card
  is not used (Spellcast3D runs the maker with it hidden).
- **When it is read.** The header's `__metadata__` records the original's name, size, modification time (ns) and
  the SHA-256 of its header, the torch version that cast it and the copy's format (`weights.FORMAT`). The loader
  reads a copy in place of its original only while that record matches the original as it is now and the running
  torch, and the copy holds exactly the tensors the model takes, in its dtypes and shapes
  (`weights.usable_copy`). A copy that is missing, stale, foreign or unreadable, or that fails while it is read,
  only means that the original is read (from its first tensor); no load fails because of a copy. A load from a
  copy casts nothing and takes no block for casts; `net_load` / `net_ready` `bytes` are what it read: the copy's
  tensors, or the original's. The read-ahead (`prefetch`) reads the copy when its record is current.
- **Which checkpoints get one.** Those where the model keeps some tensor in another dtype than the file and the
  copy is smaller: the flows. Not the decoders: the sparse-structure decoder's model keeps its file's dtypes, and
  the two others would widen fp16 tensors to fp32 (a file whose tensors all take two bytes or fewer is not even
  looked at with its model, whose class needs a card to be built). A copy that is no longer needed is removed.
- **The maker.** It skips copies that are up to date (it can run at every install), makes the missing and stale
  ones, refuses to leave less than 1 GB free on the drive, and prints one line per checkpoint with sizes and
  seconds. It exits 1 when a checkpoint got no copy, 0 otherwise. Spellcast3D runs it after its installer has
  downloaded the models and after the multi-view download (its `install/components.py` and `multiview.py`).

## Extending it

- **Another part.** Write `fn(ctx, prop, mode)` in `parts.py`, add it to `TABLE` and its key to `types.PARTS` at
  its place in the order; name its networks in `ENTER` (loaded when a group enters the part) or `LAZY` (loaded by
  the part itself), its large inputs in `WANTS`, its large outputs in `PRODUCES`, and what nobody reads after it in
  `SPENT`. Register a new network in `engine.py`.
- **Intermediate data from elsewhere.** A part is a function of the store. To start a prop from data made
  elsewhere (an edited sparse structure, a latent from another tool), put the items the next part reads into the
  store under the prop's id - the table above gives names, shapes and dtypes - and let the batch begin at that
  part; the checkpoint used by `resume` is the built-in case (`runner._restore`).
- **Reading intermediate data out.** Pass `store_factory=` to `StagedEngine` with a wrapper around `PropStore` that
  copies what the parts put (`code/spellcast/profiling/p3d_checks/check_engine.py` in Spellcast3D does this).

## How it is verified

- The logic of `engine.py`, `runner.py` and `parts.py` - order, waves, ladder, classifier, cancel, failure
  isolation, resume, the generator's hand-over, events - is tested on the CPU with everything below replaced by
  fakes (`check_engine.py --fake` in Spellcast3D's `code/spellcast/profiling/p3d_checks/`).
- The copies (`weights.py`) and the loader reading them or the original - what a copy holds and how it is
  rounded, stale, foreign and broken copies, a copy breaking half-way, the read-ahead, a pipeline folder - are
  tested on the CPU with small stand-in models (`code/tests/test_pixal3d_weights.py` in Spellcast3D).
- On the card, the same script runs the engine on golden cases: tensors captured from the untouched reference
  path stage by stage (conditionings, coords, generator states, latents, the decoded mesh). Integers must be equal;
  floats must be bit-equal wherever the reference reproduces itself bit for bit. The comparison is always against
  those captured tensors, never against a live run of the reference path: the fork's own convolutions and decoders
  changed under both, so the reference is no longer an independent control.
