"""NAF (the guided feature upsampler) one attention head at a time, read only where the caller needs it.

`naf(image, lr_features, target_size)` builds the whole upsampled map [B, C, T, T] and the caller then reads a few
thousand points of it. At T = 1024 with DINOv3's 1024 channels that call takes 12.4 GiB on the card: the map is
4 GiB, the values enlarged to its size another 4, the queries and the keys 1 GiB each in two layouts; before that,
the position rotation (RoPE) of the guide's map holds 4.9 GiB.

NAF's heads do not see each other: the rotation is element-wise with the same angles for every head, the keys are
an average pool per channel, the values are copied by a nearest resize, and the attention kernel works a head at a
time. So `naf_sampled` turns the guide's map by RoPE over itself a head at a time, then runs the key pool, the
resize and the attention for one head, hands that head's map [B, C / heads, T, T] to `sample` and keeps only what
`sample` returns. The values are those of the whole call, bit for bit. At T = 1024 a head's 256 value channels go
through the attention VALUE_SLICE at a time (each output channel is a weighted sum of its own value channel, with
weights made from the queries and keys alone), so the largest moment of a head there is the guide's map (1 GiB),
that head's queries and keys (0.25 GiB each), a slice's values and output (0.5 GiB each).
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

# Value channels per attention call. natten's cutlass-fna kernel gives a head's output channels bit for bit the
# same whether it is handed the head's 256 value channels at once or in slices of 128 (checked on an RTX 3060,
# natten 0.21.6, float32, at T = 512 and 1024: Spellcast3D code/devtools/pixal3d-lowmem/round-g/cond_hr/
# naf_slices.py); slices of 64 or 32 - no wider than the queries' 64 channels - take another kernel and round
# differently. A slice is used only where it is wider than the queries' channels and divides the head's, and only
# where a head's values would take more than VALUE_SLICE_ABOVE_MIB (T = 1024: 1 GiB; at T = 512 the whole head is
# faster and its 256 MiB are not the widest moment).
VALUE_SLICE = 128
VALUE_SLICE_ABOVE_MIB = 512
# guide_map empties PyTorch's cache before the conv stacks: under a cap, the blocks the previous view's attention
# left cut up would otherwise keep cuDNN's work area out, and cuDNN would go on with another algorithm (an
# out-of-memory event the runner counts as a failure). It costs about 0.03 s a run at 1024 px.
TRIM_BEFORE_ENCODERS = True


def _attention_of(naf) -> Optional[Callable]:
    """natten's `na2d` as NAF's own attention module holds it, or None when that module runs the two-call form of
    older natten releases (src/layers/attentions.py:6-11)."""
    module = sys.modules.get(type(naf.upsampler).__module__)
    return getattr(module, "na2d", None) if getattr(module, "NATTEN_RECENT", False) else None


def _rope_by_head(rope, x: torch.Tensor) -> torch.Tensor:
    """rope(x) for the guide's map x [B, C, H, W] when RoPE's heads are the attention's, one head at a time, written
    over x (returned) when the rotation keeps x's dtype; rope._cached_coords must be those of x's H x W.

    NAF's rotate makes the angles, cosines and sines of the whole map at every call, the same for every head
    (layers/rope.py:137-153). They are made here once, by the same operations, and each head is turned by NAF's own
    rope_apply. They are dropped before the first attention, the largest moment of the call."""
    rope_apply = sys.modules[type(rope).__module__].rope_apply
    batch, channels, height, width = x.shape
    d = channels // rope.num_heads
    angles = 2 * math.pi * rope._cached_coords[:, :, None] / rope.periods[None, None, :]
    angles = angles.flatten(1, 2)
    angles = angles.tile(2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    del angles
    dtype = torch.result_type(x, cos)
    turned = x if dtype == x.dtype else x.new_empty(x.shape, dtype=dtype)
    for i in range(rope.num_heads):
        head = rope_apply(x[:, d * i:d * (i + 1)].permute(0, 2, 3, 1).reshape(batch, 1, height * width, d), sin, cos)
        turned[:, d * i:d * (i + 1)] = head.reshape(batch, height, width, d).permute(0, 3, 1, 2)
        del head
    return turned


def _heads_last(x: torch.Tensor) -> torch.Tensor:
    """[B, d, H, W] of one head -> [B, H, W, 1, d], the layout the attention kernel takes, as one block of memory
    (natten would otherwise make that copy itself and the caller's view would keep the first one alive)."""
    return x.permute(0, 2, 3, 1).unsqueeze(3).contiguous()


def guide_input(image: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """The picture as NAF's forward hands it to its encoder: NAF's own guard, its target cross-indexed as it is
    there - a map wider than four times the one asked for is brought down first (model/naf.py:38-48). Copied
    rather than tidied, so a target that is not square behaves as it does in NAF. The picture itself when the
    guard does nothing."""
    if image.shape[-2] > 4 * target_size[0] or image.shape[-1] > 4 * target_size[1]:
        return F.interpolate(
            image,
            size=(
                min(image.shape[-2], 4 * target_size[0], 4 * target_size[1]),
                min(image.shape[-1], 4 * target_size[1], 4 * target_size[0]),
            ),
            mode="bilinear",
            align_corners=False,
        )
    return image


@torch.no_grad()
def guide_map(naf, image: torch.Tensor) -> torch.Tensor:
    """The guide's map before the pool to a target: what ImageEncoder.forward_encoder makes of the picture up to
    its last line (model/naf.py:31-35), the two convolution stacks over the whole picture - about 0.36 s a view at
    1024 px, the same whatever the target. `image` as guide_input gives it. Pooled to a target with
    `F.adaptive_avg_pool2d(map, target)`, it is forward_encoder's result for that target; at the map's own size
    the pool averages one pixel each, so the map itself is that result.

    The 3x3 stack runs first: its widest moment (a 1024 px picture: its input, the padded copy, the output and
    cuDNN's work area, about 2 GiB) then comes without the other stack's map on the card. The two stacks do not
    see each other, and the map is put together in NAF's order."""
    encoder = naf.image_encoder
    if not encoder.use_encoder:
        return image
    if TRIM_BEFORE_ENCODERS and image.is_cuda:
        torch.cuda.empty_cache()
    sem = encoder.sem_encoder(image)
    return torch.cat([encoder.encoder(image), sem], dim=1)


@torch.no_grad()
def naf_sampled(naf, image: torch.Tensor, lr_features: torch.Tensor, target_size: Tuple[int, int],
                sample: Callable[[torch.Tensor], torch.Tensor], out: Optional[torch.Tensor] = None,
                guide: Optional[torch.Tensor] = None, add: bool = False) -> torch.Tensor:
    """NAF.forward(image, lr_features, target_size) fed head by head to `sample`.

    image [B, 3, H, W] in 0..1; lr_features [B, C, h, w]; sample(fmap [B, c, T, T]) -> [B, K, c] for a slice of c
    channels of a head; returns [B, K, C] (written into `out` when given). Same operations as NAF's own code:
    model/naf.py:37-52 and 105-117, layers/rope.py:137-174, layers/attentions.py:55-74.

    `sample` is given a head's map a slice of VALUE_SLICE channels at a time (the whole head where the slice does
    not apply); on the fall-back below, where the whole map is built at once, it is given all the channels in one
    call instead. Both callers sample each channel on its own, so that makes no difference to them.

    `guide`: the guide's map at `target_size` made outside (guide_map, pooled), in place of the encoder's run
    here; it is written over by the rotation and not usable afterwards. A caller that needs the encoder's map at
    two targets makes it once this way.

    `add`: the rows are added into `out` (float32, element by element: what `out.add_(rows)` would give) instead
    of written over it, for a caller that sums views into one tensor without a copy per view.
    """
    if add and out is None:
        raise ValueError("add=True needs the tensor to add into (out=)")
    na2d = _attention_of(naf)
    if na2d is None:
        if guide is not None:
            raise ValueError("a guide made outside needs natten's na2d; this NAF runs the two-call form")
        rows = sample(naf(image, lr_features, target_size))      # an older natten: no kernel to call per head
        if out is None:
            return rows
        if add:
            out.add_(rows)
        else:
            out.copy_(rows)
        return out

    encoder, attention = naf.image_encoder, naf.upsampler
    heads = attention.num_heads
    if lr_features.shape[1] % heads:
        raise ValueError(f"NAF's {lr_features.shape[1]} channels do not divide into {heads} heads")
    if guide is None:
        # The guide's map before RoPE: forward_encoder's result (the two stacks, the pool to the target), the
        # stacks as guide_map runs them; at the map's own size the pool is the identity and is left out.
        x = guide_map(naf, guide_input(image, target_size))
        if tuple(x.shape[-2:]) != tuple(target_size):
            x = F.adaptive_avg_pool2d(x, target_size)
    else:
        if tuple(guide.shape[-2:]) != tuple(target_size):
            raise ValueError(f"the guide's map is {tuple(guide.shape[-2:])}, the target {tuple(target_size)}")
        x = guide
    batch, channels, height, width = x.shape
    if channels % heads:
        raise ValueError(f"the guide's {channels} channels do not divide into {heads} heads")

    rope = encoder.rope
    by_head = rope.num_heads == heads
    if by_head:
        if (height, width) != rope._cached_hw:
            rope._cached_coords = rope.create_coordinate(H=height, W=width)
            rope._cached_hw = (height, width)
        x = _rope_by_head(rope, x)
    else:                                                                   # RoPE's heads are not the attention's
        x = rope(x)

    d = channels // heads
    dv = lr_features.shape[1] // heads
    big = batch * height * width * dv * lr_features.element_size() > VALUE_SLICE_ABOVE_MIB << 20
    step = VALUE_SLICE if big and dv > VALUE_SLICE > d and dv % VALUE_SLICE == 0 else dv
    low = lr_features.shape[-2:]
    dilation = (height // low[0], width // low[1])
    for i in range(heads):
        xh = x[:, d * i:d * (i + 1)]
        if by_head:
            # This head's channels of rope(x) as one block of memory, as NAF's own map holds a head's channels
            # (channels first), so that the key pool runs the kernel it runs there whatever the batch.
            xh = xh.contiguous()
        q = _heads_last(xh)
        k = F.adaptive_avg_pool2d(xh, low)
        del xh
        k = _heads_last(F.interpolate(k, size=(height, width), mode="nearest-exact")).to(q.dtype)
        for c in range(dv * i, dv * (i + 1), step):        # the head's value channels, a slice at a time
            v = _heads_last(
                F.interpolate(lr_features[:, c:c + step], size=(height, width), mode="nearest-exact")
            ).to(q.dtype)
            o = na2d(q, k, v, kernel_size=attention.kernel_size, dilation=dilation, stride=1, backend="cutlass-fna")
            del v
            rows = sample(o.reshape(batch, height, width, step).permute(0, 3, 1, 2))
            del o
            if out is None:
                out = rows.new_empty(*rows.shape[:-1], dv * heads)
            if add:
                out[..., c:c + step] += rows
            else:
                out[..., c:c + step] = rows
            del rows
        del q, k
    return out
