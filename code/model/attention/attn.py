import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_
from torch.overrides import has_torch_function
from torch.nn import functional as F
from torch import nn

from typing import Optional, Tuple
import logging

from .utils import _diag_mm

_logger = logging.getLogger(__name__)


def partitioned_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    *, partitions, transpose=True
) -> Tuple[Tensor, Optional[Tensor]]:
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    assert not has_torch_function(tens_ops)

    is_batched = F._mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    assert not isinstance(embed_dim, Tensor)
    assert not use_separate_proj_weight
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
    assert query is key and query is value, "This function only supports self-attention"

    max_dim = partitions[0][1] - partitions[0][0]
    query = query.view(tgt_len, bsz, len(partitions), max_dim)
    qkv = _diag_mm(query, in_proj_weight, transpose=transpose)
    qkv += in_proj_bias
    q, k, v = qkv.chunk(3, dim=-1)

    assert attn_mask is None
    assert not (key_padding_mask is not None and key_padding_mask.dtype == torch.uint8)
    assert bias_k is None
    assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, -1).transpose(0, 1)
    assert static_k is None
    k = k.contiguous().view(k.shape[0], bsz * num_heads, -1).transpose(0, 1)
    assert static_v is None
    v = v.contiguous().view(v.shape[0], bsz * num_heads, -1).transpose(0, 1)

    assert not add_zero_attn

    src_len = k.size(1)

    assert key_padding_mask is None

    # convert mask to float
    assert not (attn_mask is not None and attn_mask.dtype == torch.bool)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, len(partitions), max_dim)
    attn_output = _diag_mm(attn_output, out_proj_weight, transpose=transpose)
    attn_output += out_proj_bias
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(-1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class PartitionedMultiheadAttention(nn.Module):
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None, *, partitions, transpose=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.partitions = partitions
        self.transpose = transpose

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self._qkv_same_embed_dim is not False
        max_dim = max(range_max - range_min for range_min, range_max in partitions)
        output_dim = max_dim * len(self.partitions)
        self.in_proj_weight = nn.Parameter(torch.empty((len(partitions), 3 * max_dim, max_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        assert bias
        self.in_proj_bias = nn.Parameter(torch.empty(3 * output_dim, **factory_kwargs))
        self.out_proj_weight = nn.Parameter(torch.empty((len(partitions), max_dim, max_dim), **factory_kwargs))
        self.out_proj_bias = nn.Parameter(torch.empty(output_dim, **factory_kwargs))

        assert not add_bias_kv
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        assert self._qkv_same_embed_dim
        [xavier_uniform_(self.in_proj_weight[i]) for i in range(self.in_proj_weight.size(0))]
        [xavier_uniform_(self.out_proj_weight[i]) for i in range(self.out_proj_weight.size(0))]

        assert self.in_proj_bias is not None
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj_bias, 0.)
        assert self.bias_k is None
        assert self.bias_v is None

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3
        assert not self.batch_first
        why_not_fast_path = "batch_first was not True"

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        assert not (self.batch_first and is_batched)

        assert self._qkv_same_embed_dim

        attn_output, attn_output_weights = partitioned_multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj_weight, self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights, partitions=self.partitions,
            transpose=self.transpose
        )

        assert not (self.batch_first and is_batched)
        return attn_output, attn_output_weights
