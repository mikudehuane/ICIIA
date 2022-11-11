# wrap a TransformerEncoder, to feed base vectors
import torch
from torch import nn


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

    def forward(self, inputs, stored_inputs):
        stored_inputs = stored_inputs.to(device=inputs.device)
        full_inputs = torch.cat([stored_inputs, inputs], dim=0)
        full_outputs = self.encoder(full_inputs)
        return full_outputs[-inputs.size(0):]
