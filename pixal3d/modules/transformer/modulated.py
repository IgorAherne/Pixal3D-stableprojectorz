from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention, ProjectAttention, GatedProjectAttention
from ..norm import LayerNorm32
from .. import fused
from .blocks import FeedForwardNet


def _fused_rows(x: torch.Tensor) -> Optional[dict]:
    """How the fused kernels find each row's batch item (x is [B, L, C]: row // L), or None when the block runs
    its eager code (the switch is off, or the features are not what the kernels take)."""
    if not fused.active(x) or x.dim() != 3 or not x.is_contiguous():
        return None
    return {"batch": x.shape[0], "rows_per_batch": x.shape[1]}


class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0), 
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        rows = _fused_rows(x)
        if rows is not None:
            h = fused.norm_modulate(x, self.norm1.eps, scale=scale_msa, shift=shift_msa, **rows)
            h = self.attn(h, phases=phases)
            x = fused.residual(x, h, gate=gate_msa, **rows)
            h = fused.norm_modulate(x, self.norm2.eps, scale=scale_mlp, shift=shift_mlp, **rows)
            h = self.mlp(h)
            return fused.residual(x, h, gate=gate_mlp, **rows)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, phases, use_reentrant=False)
        else:
            return self._forward(x, mod, phases)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
    Supports two image attention modes:
    - "cross": Standard cross-attention with image features
    - "proj": Projection-based attention with view-aligned features
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[int, int] = (1.0, 10000.0), 
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        image_attn_mode: Literal["cross", "proj", "gated_proj"] = "cross",
        proj_in_channels: Optional[int] = None,
        vae_in_channels: Optional[int] = None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.image_attn_mode = image_attn_mode
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        
        # Build cross attention based on mode
        if image_attn_mode == "cross":
            self.cross_attn = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
        elif image_attn_mode == "proj":
            _proj_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            cross_attn_block = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = ProjectAttention(cross_attn_block, channels, _proj_in)
        elif image_attn_mode == "gated_proj":
            _dino_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            _vae_in = vae_in_channels if vae_in_channels is not None else 16
            cross_attn_block = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = GatedProjectAttention(cross_attn_block, channels, _dino_in, _vae_in)
        else:
            raise ValueError(f"Unknown image attention mode: {image_attn_mode}")
            
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        else:
            self.modulation = nn.Parameter(torch.randn(6 * channels) / channels ** 0.5)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None,
                 zero: bool = False, proj_cache: bool = False) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        rows = _fused_rows(x)
        if rows is not None:
            return self._forward_fused(x, context, phases, rows, shift_msa, scale_msa, gate_msa, shift_mlp,
                                       scale_mlp, gate_mlp, zero, proj_cache)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h, phases=phases)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        if type(self.cross_attn) is ProjectAttention:
            # zero: the zero context's one row; proj_cache: proj_linear's kept answer (proj_attention.py)
            h = self.cross_attn(h, context, zero=zero, cache=proj_cache)
        else:
            h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def _forward_fused(self, x: torch.Tensor, context: Any, phases: Optional[torch.Tensor], rows: dict,
                       shift_msa: torch.Tensor, scale_msa: torch.Tensor, gate_msa: torch.Tensor,
                       shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor,
                       zero: bool = False, proj_cache: bool = False) -> torch.Tensor:
        """_forward with the elementwise ops fused (modules/fused.py: the switch PIXAL3D_FUSED); the same
        sub-layers in the same order."""
        h = fused.norm_modulate(x, self.norm1.eps, scale=scale_msa, shift=shift_msa, **rows)
        h = self.self_attn(h, phases=phases)
        x = fused.residual(x, h, gate=gate_msa, **rows)
        shape = (x.shape[0], x.shape[1])
        cached = self.cross_attn.zero_row(shape) if zero and type(self.cross_attn) is ProjectAttention else None
        if cached is not None:
            # the zero context's one row per batch item for every token (proj_attention.py); no norm2 needed
            x = fused.residual(x, cached.expand(*x.shape).contiguous())
        elif type(self.cross_attn) is ProjectAttention:
            h = fused.norm_modulate(x, self.norm2.eps, weight=self.norm2.weight, bias=self.norm2.bias, **rows)
            # ProjectAttention.forward, its sum (proj + global) done in the residual kernel
            global_context, proj_context = ((context['global'], context['proj']) if isinstance(context, dict)
                                            else context)
            global_out = self.cross_attn.cross_attn_block(h, global_context)
            proj_out = self.cross_attn.proj_feats(proj_context, proj_cache and not zero)
            if zero:
                # the sum as the residual kernel rounds it (one bf16 add), first row of each batch item
                self.cross_attn.remember_zero_row(shape, proj_out[:, :1] + global_out[:, :1])
            if proj_out.shape == x.shape and proj_out.is_contiguous() and global_out.is_contiguous():
                x = fused.residual(x, global_out, second=proj_out)
            else:
                x = fused.residual(x, (proj_out + global_out).contiguous())
            del global_out, proj_out            # freed here, as on the eager path's return: not alive in the MLP
        else:
            h = fused.norm_modulate(x, self.norm2.eps, weight=self.norm2.weight, bias=self.norm2.bias, **rows)
            h = self.cross_attn(h, context)
            x = fused.residual(x, h.contiguous())
        h = fused.norm_modulate(x, self.norm3.eps, scale=scale_mlp, shift=shift_mlp, **rows)
        h = self.mlp(h)
        return fused.residual(x, h, gate=gate_mlp, **rows)

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, phases: Optional[torch.Tensor] = None,
                zero: bool = False, proj_cache: bool = False) -> torch.Tensor:
        """`zero`: the context is all zeros (a guided sampling's unconditional pass); `proj_cache`: keep
        proj_linear's answer for the next pass with this context (proj_attention.py)."""
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, phases, zero, proj_cache,
                                                     use_reentrant=False)
        else:
            return self._forward(x, mod, context, phases, zero, proj_cache)
        