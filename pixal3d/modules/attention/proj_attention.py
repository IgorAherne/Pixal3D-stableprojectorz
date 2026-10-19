"""
View-Aligned Projection Attention Module for Pixal3D

This module implements the projection-based attention mechanism that combines
global cross-attention with view-aligned projected features.

Supports two modes:
- "proj": Standard projection (DINOv3 only), per-block proj_linear
- "gated_proj": Gated fusion of DINOv3 (semantic) + VAE (color) features
"""

from typing import *
import torch
import torch.nn as nn


class ProjectAttention(nn.Module):
    """
    Projection-based Attention Module with per-block proj_linear.
    
    Combines global cross-attention with view-aligned projected features.
    Each block owns a proj_linear that projects DINOv3 features from
    proj_in_channels (e.g. 1024) to model_channels (e.g. 1536).
    
    The module receives:
    - x: Input features from the transformer
    - context: A dict with keys:
      - 'global': Global image features, shape [B, M, ctx_channels]
      - 'proj': View-aligned projected features, shape [B, N, proj_in_channels]
    
    The output combines the cross-attention result with the projected context.
    """
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels, bias=True)
        # Zero context (the unconditional passes of a guided sampling): the branch's answer is one row per batch
        # item, the same for every token and independent of the tokens (see the sparse module's note). The first
        # zero pass of a shape [B, N] runs in full and keeps its first rows, [B, 1, C]; the next ones add them.
        self._zero_rows: Dict[Tuple[int, int], torch.Tensor] = {}
        # proj_linear's answer kept across the conditional steps of a sampling, as the sparse module does it
        # (modules/sparse/attention/proj_attention.py: the engine asks for it with `cache` where the card has
        # room and drops it after the sampling).
        self._proj_cache: Optional[Tuple[tuple, torch.Tensor]] = None

    def proj_feats(self, proj_context: torch.Tensor, cache: bool = False) -> torch.Tensor:
        """proj_linear of the projected context [B, N, D]; with `cache`, the kept answer when the context is the
        tensor it was computed for (storage address, shape, version)."""
        if not cache:
            return self.proj_linear(proj_context)
        key = (proj_context.data_ptr(), tuple(proj_context.shape), proj_context._version, proj_context.dtype)
        kept = self._proj_cache
        if kept is not None and kept[0] == key:
            return kept[1]
        out = self.proj_linear(proj_context)
        self._proj_cache = (key, out)
        return out

    def drop_proj_cache(self) -> int:
        """Forget the kept proj_linear answer; returns its bytes."""
        kept, self._proj_cache = self._proj_cache, None
        return kept[1].numel() * kept[1].element_size() if kept is not None else 0

    def zero_row(self, shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        """The branch's answer to a zero context for x of shape [B, N, C], as [B, 1, C], if a pass computed it."""
        return self._zero_rows.get(shape)

    def remember_zero_row(self, shape: Tuple[int, int], rows: torch.Tensor) -> None:
        """`rows` [B, 1, C]: what every token of each batch item got from the branch on a zero context."""
        self._zero_rows[shape] = rows.detach().clone()

    def forward(self, x: torch.Tensor, context: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                zero: bool = False, cache: bool = False) -> torch.Tensor:
        """`zero`: the context is all zeros: the kept rows serve when there are any. `cache`: keep proj_linear's
        answer for the next pass with this context."""
        shape = (x.shape[0], x.shape[1])
        cached = self.zero_row(shape) if zero else None
        if cached is not None:
            return cached.expand(x.shape[0], x.shape[1], -1)
        if isinstance(context, dict):
            global_context = context['global']
            proj_context = context['proj']
        else:
            global_context, proj_context = context

        global_out = self.cross_attn_block(x, global_context)
        proj_out = self.proj_feats(proj_context, cache and not zero)
        context_combined = proj_out + global_out
        if zero:
            self.remember_zero_row(shape, context_combined[:, :1])
        return context_combined


class GatedProjectAttention(nn.Module):
    """
    Concat-Projection Attention Module for DINOv3 (semantic) + VAE (color) features.
    
    Concatenates DINOv3 and VAE projected features and applies a single linear
    projection to model_channels. This is mathematically equivalent to two
    separate proj_linears + addition, but allows cross-dimensional interactions
    between semantic and color features through the shared weight matrix.
    
    Zero-initialized for stable training: at init, fused=0 so only global
    cross-attention contributes; color+semantic signals are gradually learned.
    
    The module receives:
    - x: Input features from the transformer
    - context: A dict with keys:
      - 'global': Global image features, shape [B, M, ctx_channels]
      - 'proj_semantic': DINOv3 projected features, shape [B, N, dino_channels]
      - 'proj_color': VAE projected features, shape [B, N, vae_channels]
    """
    def __init__(
        self,
        cross_attn_block: nn.Module,
        channels: int,
        dino_in_channels: int,
        vae_in_channels: int,
    ):
        """
        Args:
            cross_attn_block: The underlying cross-attention module
            channels: Model channels (output dimension)
            dino_in_channels: DINOv3 proj feature dimension (e.g. 1024)
            vae_in_channels: VAE latent feature dimension (e.g. 16)
        """
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(dino_in_channels + vae_in_channels, channels, bias=True)
        # Zero-init: at start, fused=0, only global cross-attn contributes
        nn.init.zeros_(self.proj_linear.weight)
        nn.init.zeros_(self.proj_linear.bias)
        
    def forward(self, x: torch.Tensor, context: Union[Dict[str, torch.Tensor], Tuple]) -> torch.Tensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_semantic = context['proj_semantic']
            proj_color = context['proj_color']
        else:
            global_context, proj_semantic, proj_color = context
        
        global_out = self.cross_attn_block(x, global_context)
        fused = self.proj_linear(torch.cat([proj_semantic, proj_color], dim=-1))
        return fused + global_out
