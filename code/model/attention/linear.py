# linear version of attention
from copy import deepcopy
from typing import Optional

from torch import nn
from torch import Tensor
from torch.nn import functional as F


class _VanillaFFLayer(nn.Module):
    def __init__(self, embed_dim, dropout=0.,
                 device=None, dtype=None, *, partitions=1):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        layer_norm_eps: float = 1e-5
        activation = F.relu

        self.embed_dim = embed_dim
        assert partitions == 1, "partitioned VanillaFF has not been implemented"

        self.linear1 = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim, embed_dim, **factory_kwargs)

        self.in_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src: Tensor):
        x = src

        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor):
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        x = self.dropout1(x)
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class VanillaFF(nn.Module):
    """单纯的前向传播网络"""

    def __init__(self, embed_dim, num_layers, dim_output, dropout=0.,
                 device=None, dtype=None, *, partitions=1) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        assert partitions == 1, "partitioned VanillaFF has not been implemented"

        forward_layer = _VanillaFFLayer(embed_dim, dropout, partitions=partitions, **factory_kwargs)
        self.layers = nn.ModuleList([deepcopy(forward_layer) for _ in range(num_layers)])

        self.activation = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(embed_dim, dim_output, **factory_kwargs)

    def forward(self, inputs):
        x = inputs

        for mod in self.layers:
            x = mod(x)

        x = self.activation(x)
        x = self.classifier(x)

        return x
