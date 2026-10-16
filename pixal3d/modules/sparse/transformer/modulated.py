from typing import *
import torch
import torch.nn as nn
from ..basic import VarLenTensor, SparseTensor
from ..attention import SparseMultiHeadAttention, SparseProjectAttention, SparseGatedProjectAttention
from ...norm import LayerNorm32
from ... import fused
from ..attention import SparseProjectAttention as _ProjectAttention
from .blocks import SparseFeedForwardNet


def _fused_rows(x: SparseTensor) -> Optional[dict]:
    """How the fused kernels find each row's batch item, or None when the block runs its eager code (the switch is
    off, or the features are not what the kernels take)."""
    feats = x.feats
    if not fused.active(feats) or feats.dim() != 2 or not feats.is_contiguous():
        return None
    batch = x.shape[0]
    if batch == 1:
        return {"batch": 1, "rows_per_batch": feats.shape[0]}
    return {"batch": batch, "batch_index": x.batch_boardcast_map}


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
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
        self.mlp = SparseFeedForwardNet(
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

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        rows = _fused_rows(x)
        if rows is not None:
            h = x.replace(fused.norm_modulate(x.feats, self.norm1.eps, scale=scale_msa, shift=shift_msa, **rows))
            h = self.attn(h)
            x = x.replace(fused.residual(x.feats, h.feats, gate=gate_msa, **rows))
            h = x.replace(fused.norm_modulate(x.feats, self.norm2.eps, scale=scale_mlp, shift=shift_mlp, **rows))
            h = self.mlp(h)
            return x.replace(fused.residual(x.feats, h.feats, gate=gate_mlp, **rows))
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    
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
        attn_mode: Literal["full", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
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
        self.self_attn = SparseMultiHeadAttention(
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
            self.cross_attn = SparseMultiHeadAttention(
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
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseProjectAttention(cross_attn_block, channels, _proj_in)
        elif image_attn_mode == "gated_proj":
            _dino_in = proj_in_channels if proj_in_channels is not None else ctx_channels
            _vae_in = vae_in_channels if vae_in_channels is not None else 16
            cross_attn_block = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.cross_attn = SparseGatedProjectAttention(cross_attn_block, channels, _dino_in, _vae_in)
        else:
            raise ValueError(f"Unknown image attention mode: {image_attn_mode}")
            
        self.mlp = SparseFeedForwardNet(
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

    def _forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor],
                 zero: bool = False) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation + mod).type(mod.dtype).chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        rows = _fused_rows(x)
        if rows is not None:
            return self._forward_fused(x, context, rows, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
                                       gate_mlp, zero)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        if zero and type(self.cross_attn) is _ProjectAttention:
            h = self.cross_attn(h, context, zero=True)      # the zero context's one row (proj_attention.py)
        else:
            h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def _forward_fused(self, x: SparseTensor, context: Union[torch.Tensor, VarLenTensor], rows: dict,
                       shift_msa: torch.Tensor, scale_msa: torch.Tensor, gate_msa: torch.Tensor,
                       shift_mlp: torch.Tensor, scale_mlp: torch.Tensor, gate_mlp: torch.Tensor,
                       zero: bool = False) -> SparseTensor:
        """_forward with the elementwise ops fused (modules/fused.py: the switch PIXAL3D_FUSED); the same
        sub-layers in the same order."""
        h = x.replace(fused.norm_modulate(x.feats, self.norm1.eps, scale=scale_msa, shift=shift_msa, **rows))
        h = self.self_attn(h)
        x = x.replace(fused.residual(x.feats, h.feats, gate=gate_msa, **rows))
        tokens = x.feats.shape[0]
        cached = self.cross_attn.zero_row(tokens) if zero and type(self.cross_attn) is _ProjectAttention else None
        if cached is not None:
            # the zero context's one row for every token (proj_attention.py); norm2 is not needed for it
            x = x.replace(fused.residual(x.feats, cached.expand(tokens, -1).contiguous()))
        elif type(self.cross_attn) is _ProjectAttention:
            h = x.replace(fused.norm_modulate(x.feats, self.norm2.eps, weight=self.norm2.weight,
                                              bias=self.norm2.bias, **rows))
            # SparseProjectAttention.forward, its sum (proj + global) done in the residual kernel
            global_context, proj_context = ((context['global'], context['proj']) if isinstance(context, dict)
                                            else context)
            global_out = self.cross_attn.cross_attn_block(h, global_context)
            proj_feats = self.cross_attn.proj_linear(proj_context.feats if isinstance(proj_context, SparseTensor)
                                                     else proj_context)
            if zero:
                # the sum as the residual kernel rounds it (one bf16 add), first row
                self.cross_attn.remember_zero_row(tokens, proj_feats[:1] + global_out.feats[:1])
            if proj_feats.shape == global_out.feats.shape and proj_feats.is_contiguous():
                x = x.replace(fused.residual(x.feats, global_out.feats, second=proj_feats))
            else:
                x = x.replace(fused.residual(x.feats, (proj_feats + global_out.feats).contiguous()))
            del global_out, proj_feats          # freed here, as on the eager path's return: not alive in the MLP
        else:
            h = x.replace(fused.norm_modulate(x.feats, self.norm2.eps, weight=self.norm2.weight,
                                              bias=self.norm2.bias, **rows))
            h = self.cross_attn(h, context)
            x = x.replace(fused.residual(x.feats, h.feats))
        h = x.replace(fused.norm_modulate(x.feats, self.norm3.eps, scale=scale_mlp, shift=shift_mlp, **rows))
        h = self.mlp(h)
        return x.replace(fused.residual(x.feats, h.feats, gate=gate_mlp, **rows))

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: Union[torch.Tensor, VarLenTensor],
                zero: bool = False) -> SparseTensor:
        """`zero`: the context is all zeros (a guided sampling's unconditional pass; proj_attention.py)."""
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, zero, use_reentrant=False)
        else:
            return self._forward(x, mod, context, zero)
