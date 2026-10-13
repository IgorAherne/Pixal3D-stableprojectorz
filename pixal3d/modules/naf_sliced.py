"""NAF (the guided feature upsampler) one attention head at a time, read only where the caller needs it.

`naf(image, lr_features, target_size)` builds the whole upsampled map [B, C, T, T] and the caller then reads a few
thousand points of it. At T = 1024 with DINOv3's 1024 channels that call takes 12.4 GiB on the card: the map is
4 GiB, the values enlarged to its size another 4, the queries and the keys 1 GiB each in two layouts; before that,
the position rotation (RoPE) of the guide's map holds 4.9 GiB.

NAF's heads do not see each other: the rotation is element-wise with the same angles for every head, the keys are
an average pool per channel, the values are copied by a nearest resize, and the attention kernel works a head at a
time. So `naf_sampled` turns the guide's map by RoPE over itself a head at a time, then runs the key pool, the
resize and the attention for one head, hands that head's map [B, C / heads, T, T] to `sample` and keeps only what
`sample` returns. The values are those of the whole call, bit for bit; the largest moment is one head's attention -
at T = 1024 the guide's map (1 GiB), that head's queries and keys (0.25 GiB each), its values and its output (1 GiB
each).
"""

from __future__ import annotations

import math
import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


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


@torch.no_grad()
def naf_sampled(naf, image: torch.Tensor, lr_features: torch.Tensor, target_size: Tuple[int, int],
                sample: Callable[[torch.Tensor], torch.Tensor], out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """NAF.forward(image, lr_features, target_size) fed head by head to `sample`.

    image [B, 3, H, W] in 0..1; lr_features [B, C, h, w]; sample(fmap [B, C / heads, T, T]) -> [B, K, C / heads];
    returns [B, K, C] (written into `out` when given). Same operations as NAF's own code: model/naf.py:37-52 and
    105-117, layers/rope.py:137-174, layers/attentions.py:55-74.

    `sample` is given one head's map; on the fall-back below, where the whole map is built at once, it is given
    all the channels in one call instead. Both callers sample each channel on its own, so that makes no
    difference to them.
    """
    na2d = _attention_of(naf)
    if na2d is None:
        rows = sample(naf(image, lr_features, target_size))      # an older natten: no kernel to call per head
        if out is None:
            return rows
        out.copy_(rows)
        return out

    encoder, attention = naf.image_encoder, naf.upsampler
    heads = attention.num_heads
    if lr_features.shape[1] % heads:
        raise ValueError(f"NAF's {lr_features.shape[1]} channels do not divide into {heads} heads")
    # NAF's own guard, its target cross-indexed as it is there: a map wider than four times the one asked for is
    # brought down first. Copied rather than tidied, so a target that is not square behaves as it does in NAF.
    if image.shape[-2] > 4 * target_size[0] or image.shape[-1] > 4 * target_size[1]:
        image = F.interpolate(
            image,
            size=(
                min(image.shape[-2], 4 * target_size[0], 4 * target_size[1]),
                min(image.shape[-1], 4 * target_size[1], 4 * target_size[0]),
            ),
            mode="bilinear",
            align_corners=False,
        )
    x = encoder.forward_encoder(image, target_size)                         # the guide's map before RoPE
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
        v = _heads_last(
            F.interpolate(lr_features[:, dv * i:dv * (i + 1)], size=(height, width), mode="nearest-exact")
        ).to(q.dtype)
        o = na2d(q, k, v, kernel_size=attention.kernel_size, dilation=dilation, stride=1, backend="cutlass-fna")
        del q, k, v
        rows = sample(o.reshape(batch, height, width, dv).permute(0, 3, 1, 2))
        del o
        if out is None:
            out = rows.new_empty(*rows.shape[:-1], dv * heads)
        out[..., dv * i:dv * (i + 1)] = rows
        del rows
    return out
