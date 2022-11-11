import time
from collections import OrderedDict
from copy import deepcopy
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .utils import _uniform_partition_integral, _event_dict, _check_partitions, DiagLinear
from .attn import PartitionedMultiheadAttention

import logging

_logger = logging.getLogger(__name__)


class OurTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, *, need_weights=False):
        x = src
        attn_result = self._sa_block(x, src_mask, src_key_padding_mask, need_weights=need_weights)
        if need_weights:
            attn_result, attn_weights = attn_result
        x = self.norm1(x + attn_result)
        x = self.norm2(x + self._ff_block(x))
        if need_weights:
            # attn_weights: (1, query_len, key_len)
            """
            F._scaled_dot_product_attention.py
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output = torch.bmm(attn, v)
            """
            # so row_idx corresponds to the current image and col_idx corresponds to the historical image
            return x, attn_weights
        else:
            return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                  *, need_weights=False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=need_weights)
        if need_weights:
            x, attn_weights = x
        else:
            x = x[0]
        x = self.dropout1(x)

        if need_weights:
            return x, attn_weights
        else:
            return x


class PartitionedTransformerEncoderLayer(nn.Module):
    def __init__(self, dim_input, *, num_heads: int, dropout: float, num_partitions: int = 1,
                 _sync=False, **kwargs):
        assert num_partitions > 0, "num_partitions must be greater than 0"
        super().__init__()
        # 只允许等大 partition，因此 base 取 1 即可
        self.partitions = _uniform_partition_integral(0, dim_input, num_partitions, base=1)
        assert _check_partitions(self.partitions), "all partitions must have the same size"

        self.encoder_layers = nn.ModuleList()
        for range_min, range_max in self.partitions:
            dim_partition = range_max - range_min
            encoder_layer = OurTransformerEncoderLayer(dim_partition, num_heads,
                                                       dim_feedforward=dim_partition, dropout=dropout)
            self.encoder_layers.append(encoder_layer)

        self.profile_results = OrderedDict()
        self._sync = _sync

    def forward(self, inputs, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, *,
                need_weights=False):
        self.profile_results.clear()
        self._profile_add("start", time.time())
        assert src_mask is None and src_key_padding_mask is None, "src_mask and key_padding_mask are not supported"
        outputs = []
        avg_weights = None
        for idx, (range_min, range_max) in enumerate(self.partitions):
            self._profile_add(idx, time.time())
            inputs_partition = inputs[:, :, range_min: range_max]
            outputs_partition = self.encoder_layers[idx](inputs_partition, src_mask, src_key_padding_mask,
                                                         need_weights=need_weights)
            if need_weights:
                outputs_partition, weights = outputs_partition
                if avg_weights is not None:
                    avg_weights += weights
                else:
                    avg_weights = weights
            outputs.append(outputs_partition)
        if avg_weights is not None:
            avg_weights = avg_weights / len(self.partitions)
        self._profile_add(len(self.partitions), time.time())
        outputs = torch.cat(outputs, dim=2)
        self._profile_add("end", time.time())
        if need_weights:
            return outputs, avg_weights
        else:
            return outputs

    def _profile_add(self, key, value):
        if self._sync:
            torch.cuda.synchronize()
        self.profile_results[key] = value


class PartitionedTransformerEncoderLayerEfficient(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, dim_input, *, num_heads: int, dropout: float, num_partitions: int = 1, transpose=True,
                 _sync=False):
        assert num_partitions > 0, "num_partitions must be greater than 0"

        layer_norm_eps: float = 1e-5
        activation = F.relu

        super().__init__()
        self.partitions = _uniform_partition_integral(0, dim_input, num_partitions, base=1)
        assert _check_partitions(self.partitions), "all partitions must have the same size"

        self.self_attn = PartitionedMultiheadAttention(dim_input, num_heads, dropout=dropout, batch_first=False,
                                                       partitions=self.partitions, transpose=transpose)
        # Implementation of Feedforward model
        self.linear1 = DiagLinear(dim_input, dim_input, num_partitions, transpose=transpose)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = DiagLinear(dim_input, dim_input, num_partitions, transpose=transpose)

        self.norm1 = nn.LayerNorm(dim_input, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_input, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        self._sync = _sync
        self.profile_results = OrderedDict()

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        self._profile_add("start", time.time())
        x = src

        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        self._profile_add("sa", time.time())
        x = self.norm2(x + self._ff_block(x))
        self._profile_add("ff", time.time())
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _profile_add(self, key, value):
        if self._sync:
            torch.cuda.synchronize()
        self.profile_results[key] = value


class NoAffineTransformerEncoderLayer(nn.Module):
    def __init__(self, dim_input, *, num_heads: int, dropout: float, _sync=False):
        layer_norm_eps: float = 1e-5
        activation = F.relu

        super().__init__()

        self.num_heads = num_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(dim_input, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)

        self.activation = activation

        self._sync = _sync
        self.profile_results = OrderedDict()

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        self._profile_add("start", time.time())
        x = src

        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        self._profile_add("sa", time.time())
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        tgt_len, bsz, embed_dim = x.shape
        x = x.contiguous().view(tgt_len, bsz * self.num_heads, -1).transpose(0, 1)

        attn_output, attn_output_weights = F._scaled_dot_product_attention(x, x, x, attn_mask, self.dropout_p)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        return self.dropout1(attn_output)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _profile_add(self, key, value):
        if self._sync:
            torch.cuda.synchronize()
        self.profile_results[key] = value


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class PartitionedTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dim_input: int, dim_output: int,
                 *, num_heads: int, dropout: float, num_partitions: int = 1,
                 encoder_layer_cls: type = PartitionedTransformerEncoderLayerEfficient,
                 transpose: bool = True,
                 _sync=False):
        super().__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output

        if num_partitions > 0:
            if encoder_layer_cls is PartitionedTransformerEncoderLayer:
                self.partitions = _uniform_partition_integral(0, dim_input, num_partitions, num_heads)
            elif encoder_layer_cls is PartitionedTransformerEncoderLayerEfficient:
                self.partitions = _uniform_partition_integral(0, dim_input, num_partitions, base=1)
            else:
                raise ValueError("encoder_layer_cls must be PartitionedTransformerEncoderLayer or "
                                 "PartitionedTransformerEncoderLayerEfficient")
            self.max_dim = max(range_max - range_min for range_min, range_max in self.partitions)
            self.is_same_dim = all(range_max - range_min == self.max_dim for range_min, range_max in self.partitions)
            self.dim_attn_output = self.max_dim * num_partitions
            if not self.is_same_dim:
                _logger.warning("dim not match, need re-computing")

            encoder_layer = encoder_layer_cls(
                self.dim_attn_output, num_heads=num_heads, dropout=dropout, num_partitions=num_partitions,
                transpose=transpose,
                _sync=_sync
            )
        else:
            assert num_partitions == 0, "num_partitions must be non-negative"
            self.dim_attn_output = dim_input
            self.is_same_dim = True
            encoder_layer = NoAffineTransformerEncoderLayer(self.dim_attn_output, num_heads=num_heads, dropout=dropout,
                                                            _sync=_sync)
        self.encoder = nn.Module()
        self.encoder.layers = _get_clones(encoder_layer, num_layers)
        del encoder_layer

        self.activation = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(self.dim_attn_output, self.dim_output)

        self._sync = _sync
        self.profile_results = OrderedDict()

    def forward(self, inputs, *, need_weights=False):
        self.profile_results.clear()
        self._profile_add("start", time.time())
        inputs = inputs.unsqueeze(1)
        tgt_len = inputs.size(0)
        bsz = inputs.size(1)
        if not self.is_same_dim:
            inputs_partitions = torch.zeros((tgt_len, bsz, len(self.partitions), self.max_dim), dtype=inputs.dtype,
                                            device=inputs.device)
            for i, (range_min, range_max) in enumerate(self.partitions):
                inputs_partitions[:, :, i, :range_max - range_min] = inputs[:, :, range_min:range_max]
            inputs = inputs_partitions.view(tgt_len, bsz, self.dim_attn_output)
        self._profile_add("pre_proc", time.time())
        avg_weights = None
        for mod in self.encoder.layers:
            if need_weights:
                inputs = mod(inputs, need_weights=need_weights)
                inputs, weights = inputs
                if avg_weights is None:
                    avg_weights = weights
                else:
                    avg_weights += weights
                    pass
            else:
                inputs = mod(inputs)
        if need_weights:
            avg_weights /= len(self.encoder.layers)
        self._profile_add("encoder", time.time())
        inputs = inputs.squeeze(1)
        self._profile_add("post_proc", time.time())
        inputs = self.activation(inputs)
        self._profile_add("activation", time.time())
        inputs = self.classifier(inputs)
        self._profile_add("classifier", time.time())
        if need_weights:
            return inputs, avg_weights
        else:
            return inputs

    def _profile_add(self, key, value):
        if self._sync:
            torch.cuda.synchronize()
        self.profile_results[key] = value

    def generate_trace_json(self, pid="Main"):
        trace_content = [
            *_event_dict("prep_proc", pid, "Main", self._profile_ts("start"), self._profile_ts("pre_proc")),
            *_event_dict("encoder", pid, "Main", self._profile_ts("pre_proc"), self._profile_ts("encoder")),
            *_event_dict("post_proc", pid, "Main", self._profile_ts("encoder"), self._profile_ts("post_proc")),
            *_event_dict("activation", pid, "Main", self._profile_ts("post_proc"), self._profile_ts("activation")),
            *_event_dict("classifier", pid, "Main", self._profile_ts("activation"), self._profile_ts("classifier")),
        ]

        for layer_idx, layer in enumerate(self.encoder.layers):
            for (prev_key, prev_val), (next_key, next_val) in zip(
                    list(layer.profile_results.items())[:-1], list(layer.profile_results.items())[1:]):
                trace_content.extend(
                    _event_dict(f"{layer_idx}_{next_key}", pid, "encoder",
                                self._profile_ts_val(prev_val), self._profile_ts_val(next_val))
                )

        return trace_content

    def _profile_ts(self, key):
        return self.profile_results[key] * 1000000 - self.profile_results["start"] * 1000000

    def _profile_ts_val(self, val):
        return val * 1000000 - self.profile_results["start"] * 1000000


