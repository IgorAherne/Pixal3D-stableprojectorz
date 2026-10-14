from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh
from o_voxel.convert import flexible_dual_grid_to_mesh
from . import chunked


DC_SLAB_ROWS = 1 << 20      # voxels whose quads are made at a time by dual_grid_faces


def dual_grid_faces(coords, intersected, split_weight, grid_size: int, slab: int = DC_SLAB_ROWS):
    """The faces o_voxel's flexible_dual_grid_to_mesh (train=False, a split weight given) makes for the voxels at
    `coords` [N, 4] int32 (batch column 0), a slab of voxels at a time: the same quads in the same order, so the
    same faces. Its hash map holds every voxel (16 bytes a voxel); a slab's edge table and lookups are all it
    adds, instead of the whole prop's (about 400 bytes a voxel beside its inputs, measured on the 1536 golden prop)."""
    from o_voxel import _C
    from o_voxel.convert.flexible_dual_grid import _init_hashmap

    device = coords.device
    offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis
    ], dtype=torch.int, device=device).unsqueeze(0)
    split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
    split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)
    grid = torch.tensor(np.array([grid_size] * 3), dtype=torch.int32, device=device)
    N = coords.shape[0]
    hashmap = _init_hashmap(grid, 2 * N, device=device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, coords, *grid.tolist())
    faces = []
    for start in range(0, N, slab):
        voxels = coords[start:start + slab, 1:]
        edge_neighbor_voxel = voxels.reshape(-1, 1, 1, 3) + offset                         # (n, 3, 4, 3)
        connected_voxel = edge_neighbor_voxel[intersected[start:start + slab]]             # (M, 4, 3)
        del edge_neighbor_voxel
        M = connected_voxel.shape[0]
        connected_voxel_hash_key = torch.cat([
            torch.zeros((M * 4, 1), dtype=torch.int, device=device),
            connected_voxel.reshape(-1, 3)
        ], dim=1)
        del connected_voxel
        connected_voxel_indices = _C.hashmap_lookup_3d_cuda(*hashmap, connected_voxel_hash_key,
                                                            *grid.tolist()).reshape(M, 4).int()
        del connected_voxel_hash_key
        connected_voxel_valid = (connected_voxel_indices != 0xffffffff).all(dim=1)
        quad_indices = connected_voxel_indices[connected_voxel_valid].int()                 # (L, 4)
        del connected_voxel_indices, connected_voxel_valid
        split_weight_ws = split_weight[quad_indices]
        split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]
        split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]
        faces.append(torch.where(
            split_weight_ws_02 > split_weight_ws_13,
            quad_indices[:, split_1],
            quad_indices[:, split_2]
        ).reshape(-1, 3))
        del split_weight_ws, split_weight_ws_02, split_weight_ws_13, quad_indices
    del hashmap
    return torch.cat(faces) if len(faces) != 1 else faces[0]


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution

    def dual_grid_fields(self, rows: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        """What the mesh extraction needs of tail rows `rows` [n, 7] float32 of the voxels at `coords` [n, 4] (the
        chunked decode's per-voxel tables): the mesh vertices - forward's dual vertices, placed in the grid as
        flexible_dual_grid_to_mesh places them -, the intersection flags and the split weights, each by the same
        operations as forward and o_voxel."""
        dual = (1 + 2 * self.voxel_margin) * F.sigmoid(rows[..., 0:3]) - self.voxel_margin
        aabb = torch.tensor(np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]), dtype=torch.float32, device=rows.device)
        grid = torch.tensor(np.array([self.resolution] * 3), dtype=torch.int32, device=rows.device)
        voxel_size = (aabb[1] - aabb[0]) / grid
        vertices = (coords[:, 1:].float() + dual) * voxel_size + aabb[0].reshape(1, 3)
        return {"vertices": vertices, "intersected": rows[..., 3:6] > 0, "quad_lerp": F.softplus(rows[..., 6:7])}
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        if not self.training and chunked.applies(self, x):
            # In boxes (chunked.py); the mesh extracted a slab of voxels at a time: the same vertices and faces.
            fields, coords, subs = chunked.forward_chunked(self, x, kwargs.get('guide_subs'), self.dual_grid_fields)
            faces = dual_grid_faces(coords, fields.pop("intersected"), fields.pop("quad_lerp"), self.resolution)
            del coords
            mesh = [Mesh(fields.pop("vertices"), faces)]
            return (mesh, subs) if kwargs.get('return_subs') else mesh
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = (1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin
            intersected = h.feats[..., 3:6] > 0
            quad_lerp = F.softplus(h.feats[..., 6:7])
            if h.shape[0] == 1:
                # One item is the whole table: taking it out of three SparseTensors would copy each of them.
                items = [(h.coords[:, 1:], vertices, intersected, quad_lerp)]
            else:
                items = ((v.coords[:, 1:], v.feats, i.feats, q.feats) for v, i, q in zip(
                    h.replace(vertices), h.replace(intersected), h.replace(quad_lerp)))
            mesh = [Mesh(*flexible_dual_grid_to_mesh(
                coords, v, i, q,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=False
            )) for coords, v, i, q in items]
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
