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
    ) -> SparseTensor:
        """`zero`: the context is all zeros (see __init__): the kept row serves when there is one."""
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

        if isinstance(proj_context, SparseTensor):
            proj_feats = self.proj_linear(proj_context.feats)
            combined_feats = proj_feats + global_out.feats
        else:
            proj_feats = self.proj_linear(proj_context)
            combined_feats = proj_feats + global_out.feats

        if zero:
            self.remember_zero_row(tokens, combined_feats[:1])
        return global_out.replace(combined_feats)


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
