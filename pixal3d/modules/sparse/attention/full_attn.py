from typing import *
import torch
from .. import VarLenTensor, SparseTensor
from .. import config
from ... import sage


__all__ = [
    'sparse_scaled_dot_product_attention',
]


def _cu_seqlens(owner: VarLenTensor, seqlen: List[int], device: torch.device) -> torch.Tensor:
    """
    The cumulative sequence lengths flash_attn takes: an int32 tensor [len(seqlen) + 1] on the device.

    They depend on the layout alone, so they are kept in the cache that `owner` shares with every
    tensor made from it by replace(): a layout is uploaded once, not at every attention call.
    """
    key = ('cu_seqlens', tuple(seqlen), device)
    sparse = isinstance(owner, SparseTensor)
    cu_seqlens = owner.get_spatial_cache(key) if sparse else owner._cache.get(key)
    if cu_seqlens is None:
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seqlen), dim=0)]).int().to(device)
        if sparse:
            owner.register_spatial_cache(key, cu_seqlens)
        else:
            owner._cache[key] = cu_seqlens
    return cu_seqlens


@overload
def sparse_scaled_dot_product_attention(qkv: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (VarLenTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, kv: Union[VarLenTensor, torch.Tensor]) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (VarLenTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, C] dense tensor containing Qs.
        kv (VarLenTensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: VarLenTensor, v: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: torch.Tensor, v: torch.Tensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: VarLenTensor, v: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"
    one_self = False     # the full self-attention of one sequence: SageAttention may take it (modules/sage.py)

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device
        owner = qkv

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device
        owner = q if isinstance(q, VarLenTensor) else kv

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device
        owner = q if isinstance(q, VarLenTensor) else k

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            one_self = s is not None and len(q_seqlen) == 1 and q_seqlen == kv_seqlen
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if one_self and config.SAGE and sage.usable(q, k, v):
        out = sage.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]
    elif config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        cu_seqlens_q = _cu_seqlens(owner, q_seqlen, device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = _cu_seqlens(owner, kv_seqlen, device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif config.ATTN == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
        cu_seqlens_q = _cu_seqlens(owner, q_seqlen, device)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = _cu_seqlens(owner, kv_seqlen, device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = _cu_seqlens(owner, kv_seqlen, device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    elif config.ATTN == 'flash_attn_4':
        if 'flash_attn_4' not in globals():
            from flash_attn.cute import flash_attn_varlen_func as flash_attn_4_varlen_func
        cu_seqlens_q = _cu_seqlens(owner, q_seqlen, device)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = _cu_seqlens(owner, kv_seqlen, device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = _cu_seqlens(owner, kv_seqlen, device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out, _ = flash_attn_4_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    elif config.ATTN == 'sdpa':
        from torch.nn.functional import scaled_dot_product_attention as _sdpa
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)   # [T, H, C] each
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)        # [T_KV, H, C] each
        # process each batch element independently (no varlen kernel needed)
        q_offs  = [0] + list(torch.cumsum(torch.tensor(q_seqlen),  dim=0).tolist())
        kv_offs = [0] + list(torch.cumsum(torch.tensor(kv_seqlen), dim=0).tolist())
        outs = []
        for i in range(len(q_seqlen)):
            qi = q[q_offs[i]:q_offs[i+1]].unsqueeze(0).permute(0, 2, 1, 3)    # [1, H, Lq,  C]
            ki = k[kv_offs[i]:kv_offs[i+1]].unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, Lkv, C]
            vi = v[kv_offs[i]:kv_offs[i+1]].unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, Lkv, C]
            out_i = _sdpa(qi, ki, vi)                                            # [1, H, Lq,  C]
            outs.append(out_i.permute(0, 2, 1, 3).squeeze(0))                   # [Lq, H, C]
        out = torch.cat(outs, dim=0)                                             # [T_Q, H, C]
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
