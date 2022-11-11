import logging
import math

import torch

import widgets
from torch import nn

_logger = logging.getLogger(__name__)


def _uniform_partition_integral(range_min, range_max, num_partitions, base: int):
    assert (range_max - range_min) % base == 0, "range_max - range_min must be divisible by base"

    num_groups = (range_max - range_min) // base
    group_partitions = widgets.uniform_partition(0, num_groups, num_partitions)

    result = [(range_min + group_min * base, range_min + group_max * base)
              for group_min, group_max in group_partitions]
    return result


def _check_partitions(partitions):
    return all(range_max - range_min == partitions[0][1] - partitions[0][0]
               for range_min, range_max in partitions)


def _event_dict(event_name, pid, tid, begin_ts, end_ts):
    return [
        {
            "name": event_name,
            "ph": "B",
            "pid": pid,
            "tid": tid,
            "ts": begin_ts,
        },
        {
            "name": event_name,
            "ph": "E",
            "pid": pid,
            "tid": tid,
            "ts": end_ts,
        }
    ]


def _diag_mm(mat1, mat2, *, transpose=True):
    """

    Args:
        mat1: (*, num_partitions, dim_input)
        mat2: (num_partitions, dim_output, dim_input)
        transpose: whether to transpose after the op
    """
    mat1 = mat1.unsqueeze(-2)
    mm_result = (mat1 * mat2).sum(dim=-1)  # (16, 1, 16, 112 * 3)
    # Transpose to let the partition dim be the last dim, such that adjacent elements are from different partitions.
    # This ensures that each head can see information from all partitions in the mha operation
    _logger.debug('transpose: %s', transpose)
    if transpose:
        mm_result = mm_result.transpose(-2, -1)
    mm_result = mm_result.flatten(-2)
    return mm_result


class DiagLinear(nn.Module):
    def __init__(self, in_features, out_features, num_partitions, *, transpose=True):
        super().__init__()

        self.transpose = transpose
        self.num_partitions = num_partitions
        assert in_features % self.num_partitions == 0, "in_features must be divisible by num_partitions"
        assert out_features % self.num_partitions == 0, "out_features must be divisible by num_partitions"

        self.dim_input = in_features // num_partitions
        self.dim_output = out_features // num_partitions

        self.weight = nn.Parameter(torch.empty((
            self.num_partitions, self.dim_output, self.dim_input), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty((out_features,), dtype=torch.float32))
        self._reset_parameters()

    def _reset_parameters(self):
        [nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5)) for i in range(self.num_partitions)]
        if self.bias is not None:
            fan_in = self.dim_input
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs = inputs.view(*inputs.shape[:-1], self.num_partitions, self.dim_input)
        output = _diag_mm(inputs, self.weight, transpose=self.transpose)
        output = output + self.bias
        return output
