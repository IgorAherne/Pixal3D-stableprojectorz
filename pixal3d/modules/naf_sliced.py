"""NAF (the guided feature upsampler) one attention head at a time, read only where the caller needs it.

`naf(image, lr_features, target_size)` builds the whole upsampled map [B, C, T, T] - 4 GiB at T = 1024 for DINOv3's
1024 channels, next to as much again for the enlarged values and 5 GiB for the position rotation of the guide - and
the caller then reads a few thousand points of it. NAF's heads do not see each other: the rotation (RoPE) is
element-wise with the same angles for every head, the keys are an average pool per channel, the values are copied
by a nearest resize, and the attention kernel works a head at a time. So `naf_sampled` runs RoPE, the key pool, the
resize and the attention for one head, hands that head's map [B, C / heads, T, T] to `sample` and keeps only what
`sample` returns. The numbers are those of the whole call; a quarter of the map exists at any time.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


def _attention_of(naf) -> Optional[Callable]:
    """natten's `na2d` as NAF's own attention module holds it, or None when that module runs the two-call form of
    older natten releases (src/layers/attentions.py:6-11)."""
    module = sys.modules.get(type(naf.upsampler).__module__)
    return getattr(module, "na2d", None) if getattr(module, "NATTEN_RECENT", False) else None


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
    """
    na2d = _attention_of(naf)
    if na2d is None:
        rows = sample(naf(image, lr_features, target_size))
        if out is None:
            return rows
        out.copy_(rows)
        return out

    encoder, attention = naf.image_encoder, naf.upsampler
    heads = attention.num_heads
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

    rope = encoder.rope
    by_head = rope.num_heads == heads
    if by_head:
        if (height, width) != rope._cached_hw:
            rope._cached_coords = rope.create_coordinate(H=height, W=width)
            rope._cached_hw = (height, width)
        coords = rope._cached_coords
    else:                                                                   # RoPE's heads are not the attention's
        x = rope(x)

    d = channels // heads
    dv = lr_features.shape[1] // heads
    low = lr_features.shape[-2:]
    dilation = (height // low[0], width // low[1])
    for i in range(heads):
        xh = x[:, d * i:d * (i + 1)]
        if by_head:
            xh = rope.rotate(xh.permute(0, 2, 3, 1).reshape(batch, 1, height * width, d), coords)
            # This head's channels of rope(x), laid out in memory as NAF's own map is (channels first), so that
            # the key pool runs the kernel it runs there whatever layout the rotation left.
            xh = xh.reshape(batch, height, width, d).permute(0, 3, 1, 2).contiguous()
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
