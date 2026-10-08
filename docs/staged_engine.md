# The staged engine

`pixal3d/staged/` runs the Pixal3D cascade in little memory. The reference pipelines
(`Pixal3DImageTo3DPipeline.run`, `Pixal3DMVImageTo3DPipeline.run_mv`) keep every network in RAM and move one after
the other to the card for each prop. The staged engine makes a whole batch of props **part by part**: every prop
goes through one part before any prop starts the next, and a part's networks are read from disk straight onto the
card when the part starts and dropped when it ends. Nothing rests in RAM, and PyTorch's reserve on the card is
capped so that it recycles its cache instead of spilling into shared memory.

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

requests = [PropRequest(id="chair", seed=11, resolution=1024, multi=False,
                        sampler_params={"ss": {...}, "shape": {...}, "tex": {...}})]
results = engine.run(requests, Path(work_dir), on_event=print, cancelled=lambda: False)
# {"chair": {"state": "done", "result": {..., "slow": [], "seconds": 187.2, "resolution": 1024, "resumed": False}}}

engine.release()        # before anything else needs the card (the reference pipeline, another tool)
engine.close()          # at exit: also stops the helper threads
```

- `run` blocks and must be called on the one thread that does all CUDA work of the process. It raises only
  `BatchBroken` (the CUDA context is unusable; restart the process). A prop that fails does not stop the others:
  its entry is `{"state": "failed", "error": {"kind", "detail"}}` with kind `input`, `mv_missing`, `oom`, `ram`,
  `disk`, `missing_file` or `error`.
- `cancelled()` is asked before every prop of every part, between the tensors of a network being loaded, at every
  sampler step and between the steps of the last part. Unfinished props come back as `{"state": "cancelled"}`. A
  prop whose file is being written is finished first.
- `work_dir` belongs to the batch. The engine empties it at the start and leaves it empty at the end.
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
| `bundle(req, images, meta)` | parts `features`, `cond_lr`, `cond_hr`, multi-view props | the views dict of `run_mv` (`images` at 512 and 1024, `camera_angle_x`, `camera_distance`, `transform_matrix`, `mesh_scale`) |
| `prop_scope(req, "ss")` | around the sparse-structure stage | a context manager yielding a dict the hook has filled when the block is left; it becomes part of the prop's notes |
| `finish(req, mesh, notes)` | part `finish` | writes the prop's file from the `DecodedMesh` (four CPU tensors) and returns what the result should carry |

`engine.sample_sparse_structure(cond, resolution, num_samples, sampler_params)` is the one attribute a caller may
wrap, exactly as the pipeline's method of that name: part `ss` calls it through the engine, positionally. A wrapper
that calls it again draws new noise from the prop's generator, and the state stored for the next stage is the one
after all draws.

### Events

`on_event(Event)` is called on the GPU thread and must not block.

| kind | meaning |
|---|---|
| `part_start` / `part_end` | a part begins / is over for the wave; `detail["props"]` lists the props in run order, `detail["wave"]` / `["waves"]` the wave |
| `prop_start` | a prop goes onto the card for this part; `detail["position"]` of `detail["of"]` |
| `step` | `step` of `total`: a sampler step, or one of the three steps of `finish` (shape decoded, texture decoded, holes filled) |
| `prop_rest` | the prop is done with this part and waits for the next |
| `prop_done` | the prop's file is written; `detail["result"]`, `detail["seconds"]` |
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
| `cond_hr` | NAF | `dino1024`, pictures, `prep`, `coords_hr` | `cond_hr`, `cond_tex` |
| `hr` | shape flow 1024 | `cond_hr`, `coords_hr`, `rng_lr` | `shape_slat`, `rng_hr` |
| `tex` | texture flow | `cond_tex`, `coords_hr`, `shape_slat`, `rng_hr` | `tex_slat`, the checkpoint |
| `finish` | shape decoder, texture decoder | `coords_hr`, `shape_slat`, `tex_slat`, `notes` | the prop's file, through the hook |

The flow networks exist once per kind (`<name>.sv` from `pipeline.json`, `<name>.mv` from `pipeline_mv.json`);
inside every part the multi-view props go first, then the networks are exchanged and the single-view props follow.
The decoders, DINOv3, NAF and background removal serve both kinds.

A part function is `fn(ctx, prop, mode)` in `parts.py`. It reads from the batch's store, computes, and writes its
results **as its last actions**, so an attempt that failed has changed nothing.

### What is kept between the parts

Everything a prop needs later lies in the batch's `PropStore`, under the prop's id:

| key | content |
|---|---|
| `request`, `notes` | the request's numbers; what the hooks noted (`ss_retries`, `ss_blocked`, ...). The finish hook and the checkpoint get the notes with `slow`: the parts that ran with the cap lifted |
| `image_<i>`, `prep` | the prepared pictures (PIL); `{"meta": ..., "camera": {...}}` |
| `dino512`, `dino1024` | `z` [V,1029,1024] / [V,4101,1024] float32: DINOv3's tokens, computed once per input size |
| `cond_ss` | `global` [1,5,1024], `proj` [1,4096,1024] bfloat16 |
| `cond_lr`, `cond_hr`, `cond_tex` | `global` [1,5,1024], `proj` [tokens,2048] bfloat16 |
| `coords_ss`, `coords_hr` | `coords` [tokens,4] int32 (batch, x, y, z), never re-ordered; `coords_hr` also `res` int32 [1] |
| `lr_slat`, `shape_slat`, `tex_slat` | `feats` [tokens,32] float32, denormalised |
| `rng_ss`, `rng_lr`, `rng_hr` | `state`: `torch.get_rng_state()` after the part's draws |
| checkpoint | `coords`, `res`, `shape_slat`, `tex_slat` and the notes, on disk: what a restarted batch resumes from |

Conditionings are stored in bfloat16, cast on the card with `.type(torch.bfloat16)`: the cast the denoisers make
of them in every pass. Items are thrown away as soon as their last reader is done (`parts.SPENT`).

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
3. `uncapped`: the cap is lifted, PyTorch may spill into shared memory as the reference does: slow, but done. The
   part is listed in the result's `slow` and a `prop_spilled` event is sent;
4. only then the prop fails with kind `oom`.

Only out-of-memory failures are tried again (`runner.classify`). Parts that run cuDNN convolutions are also
treated as failed when PyTorch counted a failed allocation without raising: cuDNN would have taken another
algorithm, which rounds differently.

Nothing is retried, closed or freed while an exception is being handled: its traceback holds the failed attempt's
tensors. The runner reduces a failure to strings, leaves the `except` block and only then acts.

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
- On the card, the same script runs the engine on golden cases: tensors captured from the untouched reference
  path stage by stage (conditionings, coords, generator states, latents, the decoded mesh). Integers must be equal;
  floats must be bit-equal wherever the reference reproduces itself bit for bit.
