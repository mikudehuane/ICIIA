from typing import Tuple, Iterable

import thop
import torch
from torch import nn, Tensor
from .attention import DiagLinear, PartitionedMultiheadAttention, NoAffineTransformerEncoderLayer


def extended_profile(module: nn.Module, inputs: Iterable[Tensor], **kwargs) -> Tuple[int, int]:
    from thop.vision.basic_hooks import count_convNd, zero_ops
    from efficientnet_pytorch.utils import Conv2dStaticSamePadding, MemoryEfficientSwish
    from torch.nn import Identity

    def count_layer_normalization(m: nn.LayerNorm, x, y):
        x = x[0]
        # bn is by default fused in inference
        flops = torch.DoubleTensor([2 * x.numel()])
        if m.elementwise_affine:
            flops *= 2
        m.total_ops += flops

    def count_multi_head_attention(m: nn.MultiheadAttention, x, y):
        query, key, value = x
        # in project
        batch_size = query.size(0)
        feature_dim = query.size(2) + key.size(2) + value.size(2)
        assert m.in_proj_weight.size(0) == feature_dim
        m.total_ops += torch.DoubleTensor([batch_size * feature_dim * m.in_proj_weight.size(1)])
        # attention
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0) * query.size(2)])  # Q * K
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0)])  # scale
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0)])  # softmax
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0) * query.size(2)])  # weight * V
        # out project
        m.total_ops += torch.DoubleTensor([batch_size * value.size(2) * y[0].size(2)])

    def count_na_mha(m: NoAffineTransformerEncoderLayer, x, y):
        x, = x
        m.total_ops += torch.DoubleTensor([x.size(0) * x.size(0) * x.size(2)])  # Q * K
        m.total_ops += torch.DoubleTensor([x.size(0) * x.size(0)])  # scale
        m.total_ops += torch.DoubleTensor([x.size(0) * x.size(0)])  # softmax
        m.total_ops += torch.DoubleTensor([x.size(0) * x.size(0) * x.size(2)])  # weight * V
        count_layer_normalization(m.norm1, x, y)
        m.total_ops += m.norm1.total_ops

    def count_pmha(m: PartitionedMultiheadAttention, x, y):
        query, key, value = x
        # in project
        m.total_ops += torch.DoubleTensor([query.numel() * m.in_proj_weight.size(1)])
        # attention
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0) * query.size(2)])  # Q * K
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0)])  # scale
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0)])  # softmax
        m.total_ops += torch.DoubleTensor([query.size(0) * query.size(0) * query.size(2)])  # weight * V
        # out project
        m.total_ops += torch.DoubleTensor([query.numel() * m.out_proj_weight.size(1)])

    def count_diag_linear(m: DiagLinear, x, y):
        total_ops = x[0].numel()
        total_ops *= m.weight.size(1)  # out_dim
        m.total_ops += torch.DoubleTensor([total_ops])

    def count_embedding(m: nn.Embedding, x, y):
        pass  # no flops

    custom_ops = {
        Conv2dStaticSamePadding: count_convNd,
        Identity: zero_ops,
        MemoryEfficientSwish: zero_ops,
        nn.LayerNorm: count_layer_normalization,
        nn.MultiheadAttention: count_multi_head_attention,
        DiagLinear: count_diag_linear,
        PartitionedMultiheadAttention: count_pmha,
        NoAffineTransformerEncoderLayer: count_na_mha,
        nn.Embedding: count_embedding,
    }

    results = thop.profile(module, inputs, custom_ops=custom_ops, **kwargs)

    def remove_hooks(m: nn.Module):
        if 'total_ops' in m._buffers:
            m._buffers.pop("total_ops")
        if 'total_params' in m._buffers:
            m._buffers.pop("total_params")

    module.apply(remove_hooks)
    return results
