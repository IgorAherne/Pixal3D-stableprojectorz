"""The neighbour cache of a flex_gemm sparse convolution with only what a forward pass reads.

flex_gemm builds a level's cache once (SubMConv3dFunction._compute_neighbor_cache) with everything a backward
pass will need: beside the neighbour map [n, V] it makes two transposed copies of it, a running sum of one of
them and two tables of the valid pairs, about 580 bytes per voxel while it is built and 240 kept. The forward
kernels of the masked algorithms read three things: `neighbor_map`, and `gray_code` and `sorted_idx`, from
which the cache derives the kernel offsets of every block of rows. `build` makes exactly those three, about
150 bytes per voxel while it works and 120 kept:

    neighbor_map   by the same C function, from the same hash map
    gray_code      bit v set where the voxel has a neighbour at kernel offset v
    sorted_idx     argsort of the binary number whose Gray code that is

as flex_gemm's kernel defines them (kernels/cuda/spconv/neighbor_map.cu, the gray / binary code kernel and the
argsort after it), so the three tensors are equal to flex_gemm's own and every convolution sums in the same order.

Without gradients only: the cache has none of the backward pass's tables.
"""

from typing import Optional, Tuple

import torch
from flex_gemm import kernels
from flex_gemm.ops import spconv, utils
from flex_gemm.ops.spconv import Algorithm
from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dNeighborCache

from . import codes

MASKED = (Algorithm.MASKED_IMPLICIT_GEMM, Algorithm.MASKED_IMPLICIT_GEMM_SPLITK)
MAX_KERNEL_VOLUME = 31      # the codes are kept as int32, as flex_gemm stores them: bit 31 would be the sign


def build(coords: torch.Tensor, shape: torch.Size, kernel_size: Tuple[int, int, int],
          dilation: Tuple[int, int, int]) -> Optional[SubMConv3dNeighborCache]:
    """
    The forward-only neighbour cache of the voxels `coords` [n, 4] int32 (batch, x, y, z) in a tensor of
    `shape` (N, C, W, H, D), for flex_gemm's algorithm in use; `kernel_size` and `dilation` in the order of
    the weight's kernel dimensions.

    None when this cache does not apply (gradients on, another algorithm, coords not on the card, a kernel
    of more than 31 offsets): flex_gemm then builds its own. The caller decides whether to ask at all
    (conv/config.py LEAN_NEIGHBOR_CACHE).
    """
    volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
    if (torch.is_grad_enabled() or spconv.ALGORITHM not in MASKED or not coords.is_cuda
            or volume > MAX_KERNEL_VOLUME):
        return None
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert coords.dtype == torch.int32, "Unsupported coords dtype. Expect int32"
    N, C, W, H, D = shape

    hashmap_keys, hashmap_vals = utils.init_hashmap(shape, int(spconv.HASHMAP_RATIO * coords.shape[0]), coords.device)
    neighbor_map = kernels.cuda.hashmap_build_submanifold_conv_neighbour_map_cuda(
        hashmap_keys, hashmap_vals, coords,
        W, H, D,
        kernel_size[0], kernel_size[1], kernel_size[2],
        dilation[0], dilation[1], dilation[2],
    )
    del hashmap_keys, hashmap_vals

    # A missing neighbour is 0xffffffff, which reads as -1 through an int32 view of the same memory. The Gray code
    # (bit v: a neighbour at offset v) and its binary number - the code xor-ed with itself shifted right by 1, 2,
    # ..., volume - 1 - in one Triton pass (codes.py, the same integers as the eager loops, which it falls back to).
    gray_code, binary_code = codes.codes(neighbor_map, volume)
    sorted_idx = torch.argsort(binary_code)

    return SubMConv3dNeighborCache(neighbor_map=neighbor_map, gray_code=gray_code, sorted_idx=sorted_idx)
