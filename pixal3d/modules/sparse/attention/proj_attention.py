"""
Sparse View-Aligned Projection Attention Module for Pixal3D

Sparse versions of ProjectAttention and GatedProjectAttention.

Supports two modes:
- "proj": Standard projection (DINOv3 only)
- "gated_proj": Gated fusion of DINOv3 (semantic) + VAE (color) features
"""

from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor, VarLenTensor


class SparseProjectAttention(nn.Module):
    """
    Sparse Projection-based Attention Module with per-block proj_linear.
    """
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels, bias=True)
        # Zero context (the unconditional passes of a guided sampling: `zero` below): the global context's keys
        # and values are then to_kv's bias, the same for every key, so the attention is a uniform average that
        # returns exactly that value row whatever the query; to_out of identical rows gives identical rows and
        # proj_linear of zeros its bias. Every token therefore gets the one row, which depends on the weights
        # alone, not on the tokens. The branch runs in full on the first zero pass of a token count (that pass is
        # exactly what it was) and its first row is kept here, keyed by the token count (cuBLAS may pick another
        # kernel for another count); the next zero passes of that count add the kept row instead (-10 % of a
        # detail pass). Proven bit-identical by the golden checks, which sample with guidance.
        self._zero_rows: Dict[int, torch.Tensor] = {}
        # Conditional passes: proj_linear's input, the projected conditioning, is the same tensor at every step of
        # a sampling, so its answer is too (one GEMM of one shape on one input gives the same bits each time).
        # When the engine says the card has room for the 30 answers (`cache` below: parts.proj_cache_fits), the
        # answer is kept with its input's identity (storage address, shape, version) and given again while the
        # input is that tensor; the engine drops it after the sampling (drop_proj_caches). 157 ms of a 3.1 s
        # detail pass at 21.5k tokens; 1.9 GB on the card there.
        self._proj_cache: Optional[Tuple[tuple, torch.Tensor]] = None

    def proj_feats(self, proj_context: Union[torch.Tensor, SparseTensor], cache: bool = False) -> torch.Tensor:
        """proj_linear of the projected context's rows; with `cache`, the kept answer when the context is the
        tensor it was computed for (see __init__)."""
        feats = proj_context.feats if isinstance(proj_context, SparseTensor) else proj_context
        if not cache:
            return self.proj_linear(feats)
        key = (feats.data_ptr(), tuple(feats.shape), feats._version, feats.dtype)
        kept = self._proj_cache
        if kept is not None and kept[0] == key:
            return kept[1]
        out = self.proj_linear(feats)
        self._proj_cache = (key, out)
        return out

    def drop_proj_cache(self) -> int:
        """Forget the kept proj_linear answer; returns its bytes."""
        kept, self._proj_cache = self._proj_cache, None
        return kept[1].numel() * kept[1].element_size() if kept is not None else 0

    def zero_row(self, tokens: int) -> Optional[torch.Tensor]:
        """The branch's answer to a zero context for `tokens` tokens, [1, C], if a pass computed it already."""
        return self._zero_rows.get(tokens)

    def remember_zero_row(self, tokens: int, row: torch.Tensor) -> None:
        """`row` [1, C]: what every one of `tokens` tokens got from the branch on a zero context."""
        self._zero_rows[tokens] = row.detach().clone()

    def forward(
        self,
        x: SparseTensor,
        context: Union[Dict[str, Union[torch.Tensor, VarLenTensor, SparseTensor]],
                       Tuple[Union[torch.Tensor, VarLenTensor], SparseTensor]],
        zero: bool = False,
        cache: bool = False,
    ) -> SparseTensor:
        """`zero`: the context is all zeros (see __init__): the kept row serves when there is one. `cache`: keep
        proj_linear's answer for the next pass with this context (see __init__)."""
        tokens = x.feats.shape[0]
        cached = self.zero_row(tokens) if zero else None
        if cached is not None:
            return x.replace(cached.expand(tokens, -1))
        if isinstance(context, dict):
            global_context = context['global']
            proj_context = context['proj']
        else:
            global_context, proj_context = context

        global_out = self.cross_attn_block(x, global_context)

        proj_feats = self.proj_feats(proj_context, cache and not zero)
        combined_feats = proj_feats + global_out.feats

        if zero:
            self.remember_zero_row(tokens, combined_feats[:1])
        return global_out.replace(combined_feats)


def drop_proj_caches(model: nn.Module) -> int:
    """Every kept proj_linear answer of `model`'s projection-attention modules (sparse or dense) dropped; the
    bytes freed. Anything without modules (a fake network of the tests) is left alone."""
    modules = getattr(model, "modules", None)
    if modules is None:
        return 0
    freed = 0
    for module in modules():
        if hasattr(module, "_proj_cache") and callable(getattr(module, "drop_proj_cache", None)):
            freed += module.drop_proj_cache()
    return freed


def proj_cache_bytes(model: nn.Module, tokens: int) -> int:
    """Bytes the kept proj_linear answers of `model` would take for a prop of `tokens` tokens (one [tokens, C]
    table in the layer's dtype per projection-attention module); 0 for a network without such modules."""
    modules = getattr(model, "modules", None)
    if modules is None:
        return 0
    total = 0
    for module in modules():
        if not hasattr(module, "_proj_cache"):
            continue
        linear = module.proj_linear
        width = getattr(linear, "out_features", None)
        if width is None:
            continue
        dtype = getattr(linear, "dtype", None)
        if not isinstance(dtype, torch.dtype):
            weight = getattr(linear, "weight", None)
            dtype = weight.dtype if weight is not None else torch.bfloat16
        total += tokens * int(width) * torch.empty((), dtype=dtype).element_size()
    return total


class SparseGatedProjectAttention(nn.Module):
    """
    Sparse Concat-Projection Attention Module for DINOv3 + VAE features.
    
    Concatenates DINOv3 and VAE projected features and applies a single linear
    projection to model_channels. Zero-initialized for stable training.
    
    Context dict must contain:
    - 'global': Global image features for cross-attention
    - 'proj_semantic': DINOv3 projected features (SparseTensor or Tensor)
    - 'proj_color': VAE projected features (SparseTensor or Tensor)
    """
    def __init__(
        self,
        cross_attn_block: nn.Module,
        channels: int,
        dino_in_channels: int,
        vae_in_channels: int,
    ):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(dino_in_channels + vae_in_channels, channels, bias=True)
        # Zero-init: at start, fused=0, only global cross-attn contributes
        nn.init.zeros_(self.proj_linear.weight)
        nn.init.zeros_(self.proj_linear.bias)

    def _get_feats(self, t):
        return t.feats if isinstance(t, SparseTensor) else t

    def forward(
        self,
        x: SparseTensor,
        context: Union[Dict[str, Union[torch.Tensor, VarLenTensor, SparseTensor]], Tuple],
    ) -> SparseTensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_semantic = context['proj_semantic']
            proj_color = context['proj_color']
        else:
            global_context, proj_semantic, proj_color = context

        global_out = self.cross_attn_block(x, global_context)

        fused = self.proj_linear(torch.cat([
            self._get_feats(proj_semantic),
            self._get_feats(proj_color),
        ], dim=-1))
        combined_feats = fused + global_out.feats

        return global_out.replace(combined_feats)
