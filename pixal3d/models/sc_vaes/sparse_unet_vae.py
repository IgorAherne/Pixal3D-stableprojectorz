from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module
from ...modules import sparse as sp
from ...modules.norm import LayerNorm32
from . import chunked


NORM_ROWS = 2 ** 18     # a table above this many rows is normalised a slab of rows at a time


def norm_rows(norm: LayerNorm32, feats: torch.Tensor) -> torch.Tensor:
    """
    `norm(feats)` for a table of any size.

    LayerNorm32 makes a float32 copy and a float32 result of the whole table it is given. Every row is
    normalised on its own, so a big table goes through a slab of rows at a time, straight into a result of
    the table's dtype. The table is normalised whole when gradients are wanted or autocast is on.
    """
    if feats.shape[0] <= NORM_ROWS or torch.is_grad_enabled() or torch.is_autocast_enabled():
        return norm(feats)
    out = torch.empty(feats.shape, dtype=feats.dtype, device=feats.device)   # as LayerNorm32 returns it: rows
    for start in range(0, feats.shape[0], NORM_ROWS):                         # laid out one after the other
        out[start:start + NORM_ROWS] = norm(feats[start:start + NORM_ROWS])
    return out


def cast_norm_rows(feats: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    `F.layer_norm(feats.type(dtype))` over the channels, for a table of any size.

    A big table is cast and normalised a slab of rows at a time (every row is normalised on its own), so the
    cast copy of the whole table never exists beside the result. The table is cast and normalised whole when
    gradients are wanted or autocast is on (autocast runs layer_norm in its own dtype, which a slab written
    into a result of `dtype` would narrow).
    """
    if feats.shape[0] <= NORM_ROWS or torch.is_grad_enabled() or torch.is_autocast_enabled():
        feats = feats.type(dtype)
        return F.layer_norm(feats, feats.shape[-1:])
    out = torch.empty(feats.shape, dtype=dtype, device=feats.device)
    for start in range(0, feats.shape[0], NORM_ROWS):
        rows = feats[start:start + NORM_ROWS].type(dtype)
        out[start:start + NORM_ROWS] = F.layer_norm(rows, rows.shape[-1:])
    return out


class SiLUOverInput(nn.SiLU):
    """
    SiLU that overwrites its input when gradients are off: the same values without a second table of that
    size. For right after a layer whose output nobody else holds, such as the first Linear of an MLP.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.silu(input, inplace=not torch.is_grad_enabled())


def add_into(h: sp.SparseTensor, other: sp.SparseTensor) -> sp.SparseTensor:
    """
    `h + other`, feature rows added one to one. Without gradients the sum is written into h's own table
    instead of a third one, so h must be a result nobody else holds (a convolution's output).
    """
    if torch.is_grad_enabled() or h.feats.dtype != other.feats.dtype or h.feats.shape != other.feats.shape:
        return h + other
    h.feats.add_(other.feats)
    return h


def forget_neighbors(x: sp.SparseTensor) -> None:
    """
    Drop the convolutions' neighbour tables of x's level (the flex_gemm backend keeps them in the level's
    spatial cache, shared by every tensor of the level). For after the level's last convolution, when
    gradients are off; a convolution that came later would build them again.

    The level's cache holds other things under keys of other types (the attention's sequence lengths are
    kept there under a tuple), so only the string keys are examined.
    """
    level = x.get_spatial_cache()
    for key in [key for key in level
                if isinstance(key, str) and key.startswith('SubMConv3d_neighbor_cache')]:
        del level[key]


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
        resample_mode: Literal['nearest', 'spatial2channel'] = 'nearest',
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.resample_mode = resample_mode
        self.use_checkpoint = use_checkpoint
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        if resample_mode == 'nearest':
            self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        elif resample_mode =='spatial2channel' and self.downsample:
            self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        if resample_mode == 'nearest':
            self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        elif resample_mode =='spatial2channel' and self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        elif resample_mode =='spatial2channel' and not self.downsample:
            self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        self.updown = None
        if self.downsample:
            if resample_mode == 'nearest':
                self.updown = sp.SparseDownsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseSpatial2Channel(2)
        elif self.upsample:
            self.to_subdiv = sp.SparseLinear(channels, 8)
            if resample_mode == 'nearest':
                self.updown = sp.SparseUpsample(2)
            elif resample_mode =='spatial2channel':
                self.updown = sp.SparseChannel2Spatial(2)

    def _updown(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.downsample:
            x = self.updown(x)
        elif self.upsample:
            x = self.updown(x, subdiv.replace(subdiv.feats > 0))
        return x

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        subdiv = None
        if self.upsample:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        if self.resample_mode == 'spatial2channel':
            h = self.conv1(h)
        h = self._updown(h, subdiv)
        x = self._updown(x, subdiv)
        if self.resample_mode == 'nearest':
            h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.upsample:
            return h, subdiv
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = sp.SparseDownsample(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.updown(h)
        x = self.updown(x)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        if self.pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseUpsample(2)

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        h = self.updown(h, subdiv_binarized)
        x = self.updown(x, subdiv_binarized)
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = sp.SparseSpatial2Channel(2)

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseResBlockC2S3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.pred_subdiv = pred_subdiv
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels * 8, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.repeat_interleave(out_channels // (channels // 8), dim=1))
        if pred_subdiv:
            self.to_subdiv = sp.SparseLinear(channels, 8)
        self.updown = sp.SparseChannel2Spatial(2)
        self.lean = False       # set through the decoder's `lean`

    def trim(self) -> None:
        """
        With `lean` set, give PyTorch's unused blocks back to the driver. This block goes from the coarse
        level's tables to ones four times their size: a fine table carved out of a block a coarse one left
        behind strands the rest of that block, and under a memory cap the decode then fails with memory to
        spare. No value depends on it.
        """
        if self.lean:
            torch.cuda.empty_cache()

    def _forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.pred_subdiv:
            subdiv = self.to_subdiv(x)
        h = x.replace(norm_rows(self.norm1, x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        if not torch.is_grad_enabled():
            forget_neighbors(x)     # conv1 is the last convolution on the coarse voxels
        self.trim()
        subdiv_binarized = subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        fine = self.updown.fine_level(h, subdiv_binarized)     # x is on h's voxels: one fine level serves both
        h = self.updown(h, fine=fine)
        self.trim()
        x = self.updown(x, fine=fine)
        del fine
        self.trim()
        h = h.replace(norm_rows(self.norm2, h.feats))
        h = h.replace(F.silu(h.feats))
        self.trim()
        h = self.conv2(h)
        h = add_into(h, self.skip_connection(x))
        if self.pred_subdiv:
            return h, subdiv
        else:
            return h
    
    def forward(self, x: sp.SparseTensor, subdiv: sp.SparseTensor = None) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, subdiv, use_reentrant=False)
        else:
            return self._forward(x, subdiv)
        
    
class SparseConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        
        self.norm = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.conv = sp.SparseConv3d(channels, channels, 3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            SiLUOverInput(),
            zero_module(nn.Linear(int(channels * mlp_ratio), channels)),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.conv(x)
        h = h.replace(norm_rows(self.norm, h.feats))
        h = h.replace(self.mlp(h.feats))
        return h + x
    
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseUnetVaeEncoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = sp.SparseLinear(in_channels, model_channels[0])
        self.to_latent = sp.SparseLinear(model_channels[-1], 2 * latent_channels)
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[down_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        **block_args[i],
                    )
                )
                
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: sp.SparseTensor, sample_posterior=False, return_raw=False):
        h = self.input_layer(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                h = block(h)
        h = h.type(x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.to_latent(h)
        
        # Sample from the posterior distribution
        mean, logvar = h.feats.chunk(2, dim=-1)
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        z = h.replace(z)
            
        if return_raw:
            return z, mean, logvar
        else:
            return z
    
    
class SparseUnetVaeDecoder(nn.Module):
    """
    Sparse Swin Transformer Unet VAE model.
    """
    def __init__(
        self,
        out_channels: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
        pred_subdiv: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.use_fp16 = use_fp16
        self.pred_subdiv = pred_subdiv
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.lean = False       # True: PyTorch's cache is emptied between the levels (see trim)
        # > 0: without gradients, the levels after R/4 are decoded in boxes of about this many fine voxels,
        # the same values in a memory that does not grow with the prop (chunked.py). 0: the loop below.
        self.chunk_voxels = 0
        
        self.output_layer = sp.SparseLinear(model_channels[-1], out_channels)
        self.from_latent = sp.SparseLinear(latent_channels, model_channels[0])
        
        self.blocks = nn.ModuleList([])
        for i in range(len(num_blocks)):
            self.blocks.append(nn.ModuleList([]))
            for j in range(num_blocks[i]):
                self.blocks[-1].append(
                    globals()[block_type[i]](
                        model_channels[i],
                        **block_args[i],
                    )
                )
            if i < len(num_blocks) - 1:
                self.blocks[-1].append(
                    globals()[up_block_type[i]](
                        model_channels[i],
                        model_channels[i+1],
                        pred_subdiv=pred_subdiv,
                        **block_args[i],
                    )
                )
                    
        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
            
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    @property
    def lean(self) -> bool:
        """
        Whether PyTorch's cache is emptied between the levels and between the steps of the up-blocks (see
        trim). Off by default; the staged engine sets it.
        """
        return self._lean

    @lean.setter
    def lean(self, value: bool) -> None:
        self._lean = bool(value)
        for module in self.modules():
            if module is not self and hasattr(module, 'lean'):
                module.lean = self._lean

    def trim(self) -> None:
        """
        With `lean` set, give PyTorch's unused blocks back to the driver. Called between the levels: each
        level's tables are several times larger than the last one's, and under a memory cap they would have
        to be carved out of the smaller blocks the level before left behind. No value depends on it.
        """
        if self.lean:
            torch.cuda.empty_cache()

    def forward(self, x: sp.SparseTensor, guide_subs: Optional[List[Union[sp.SparseTensor, torch.Tensor]]] = None, return_subs: bool = False) -> sp.SparseTensor:
        assert guide_subs is None or self.pred_subdiv == False, "Only decoders with pred_subdiv=False can be used with guide_subs"
        assert return_subs == False or self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with return_subs"
        if chunked.applies(self, x):
            fields, coords, subs = chunked.forward_chunked(self, x, guide_subs, chunked.raw_fields)
            h = sp.SparseTensor(fields.pop("feats"), coords)
            return (h, subs) if return_subs else h
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(self.blocks):
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if self.pred_subdiv:
                        if self.training:
                            subs_gt.append(h.get_spatial_cache('subdivision'))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        guide = guide_subs[i] if guide_subs is not None else None
                        if isinstance(guide, torch.Tensor):
                            # A bare table [n, 8] (logits or booleans), rows in the order of this level's voxels.
                            guide = h.replace(guide)
                        h = block(h, subdiv=guide)
                    self.trim()
                else:
                    h = block(h)
        if not torch.is_grad_enabled():
            forget_neighbors(h)     # nothing after the blocks convolves
            self.trim()
        h = h.replace(cast_norm_rows(h.feats, x.dtype))
        h = self.output_layer(h)
        if self.training and self.pred_subdiv:
            return h, subs_gt, subs
        else:
            if return_subs:
                return h, subs
            else:
                return h
    
    @staticmethod
    def child_coords(block: nn.Module, x: sp.SparseTensor) -> torch.Tensor:
        """
        The coords the up-block `block` would give its output for the input x, rows in the same order.
        They follow from the block's subdivision of x alone, so only that is computed: the subdivision
        table is sent through the block's own upsampling in place of the features.
        """
        subdiv = block.to_subdiv(x)
        subdiv = subdiv.replace(subdiv.feats > 0)
        return block.updown(subdiv, subdiv).coords

    def upsample(self, x: sp.SparseTensor, upsample_times: int) -> torch.Tensor:
        assert self.pred_subdiv == True, "Only decoders with pred_subdiv=True can be used with upsampling"
        
        h = self.from_latent(x)
        h = h.type(self.dtype)
        for i, res in enumerate(self.blocks):
            if i == upsample_times:
                return h.coords
            for j, block in enumerate(res):
                if i < len(self.blocks) - 1 and j == len(res) - 1:
                    if i == upsample_times - 1:
                        return self.child_coords(block, h)
                    h, sub = block(h)
                    self.trim()
                else:
                    h = block(h)
       