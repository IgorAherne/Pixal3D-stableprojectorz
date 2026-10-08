"""The end of a generation, one step at a time: latents -> mesh, latents -> texture voxels, holes closed.

The reference pipelines' decode_latent and the staged engine's last part run the same functions, so both get the
same values. Nothing is kept on the card between the steps beyond what the next one reads:

    mesh, subs = decode_shape(shape_decoder, shape_feats, coords, res)
    tex_coords, attrs = decode_texture(tex_decoder, tex_feats, coords, subs)
    fill_holes(mesh)

`feats` are the denormalised latents [N, 32] float32, `coords` their [N, 4] int32 rows (batch, x, y, z) of one
prop (batch column all 0), both on the decoder's device. The decoders are used as they are: whoever calls decides
which of them is on the card, and sets `decoder.lean = True` to keep PyTorch's reserve close to what the decode
really holds: the cache is then emptied between the levels, and the results are given blocks of their own size.

`upsample` is the step between the two shape stages: the LR latent's tokens -> the occupied voxels of the 512 grid.
"""

from typing import TYPE_CHECKING, List, Tuple

import torch

from ..modules.sparse import SparseTensor

if TYPE_CHECKING:
    from ..representations import Mesh

UPSAMPLE_TIMES = 4          # 32 grid -> 512 grid


def settle(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    The same tensors, each in a block of its own size.

    What a decode returns was allocated while PyTorch's cache held the decoder's far larger blocks: each
    result sits in a corner of one of them and keeps the whole block reserved for as long as it lives (1.5 GiB
    for the 1536 golden prop's mesh, which the texture decode then cannot use). The results are copied to the
    CPU, the cache is emptied, and they are copied back. `tensors` must hold the only references to them; the
    list is emptied.
    """
    device = tensors[0].device
    host = [tensor.cpu() for tensor in tensors]
    tensors.clear()
    torch.cuda.empty_cache()
    return [tensor.to(device) for tensor in host]


@torch.no_grad()
def decode_shape(decoder, feats: torch.Tensor, coords: torch.Tensor, res: int) -> Tuple["Mesh", List[torch.Tensor]]:
    """The shape latent decoded at resolution `res`: the mesh before its holes are closed (vertices [Vn, 3]
    float32, faces [F, 3] int32) and the four subdivision tables [n_i, 8] bool, one per level, which is all the
    texture decoder needs of this decode."""
    decoder.set_resolution(res)
    meshes, subs = decoder(SparseTensor(feats, coords), return_subs=True)
    mesh, subs = meshes[0], [sub.feats > 0 for sub in subs]
    del meshes
    if getattr(decoder, 'lean', False):
        held = [mesh.vertices, mesh.faces, *subs]
        mesh.vertices = mesh.faces = None
        del subs
        mesh.vertices, mesh.faces, *subs = settle(held)
    return mesh, subs


@torch.no_grad()
def decode_texture(decoder, feats: torch.Tensor, coords: torch.Tensor,
                   subs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The texture latent decoded along the shape's subdivision tables: the voxels' coords [N4, 3] int32
    (x, y, z) and their attributes [N4, 6] float32 in 0..1, rows in the order of the shape decoder's voxels."""
    voxels = decoder(SparseTensor(feats, coords), guide_subs=subs)
    held = [voxels.coords[:, 1:].contiguous(), voxels.feats * 0.5 + 0.5]
    del voxels
    if getattr(decoder, 'lean', False):
        held = settle(held)
    return held[0], held[1]


@torch.no_grad()
def fill_holes(mesh: "Mesh") -> None:
    """Close the small holes of the decoded mesh, in place. PyTorch's cache is emptied first: the hole filling
    allocates outside it, and the decoders have just left it at its largest."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mesh.fill_holes()


@torch.no_grad()
def upsample(decoder, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """The LR shape latent's tokens (32 grid) subdivided four times by the shape decoder: the occupied voxels
    of the 512 grid, [M, 4] int32, in the decoder's row order."""
    return decoder.upsample(SparseTensor(feats, coords), upsample_times=UPSAMPLE_TIMES)
