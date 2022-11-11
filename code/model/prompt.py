import torch
from torch import nn


class PromptFF(nn.Module):
    """linear model with prompt tuning
    """
    def __init__(self, dim_input: int, dim_prompt: int, dim_output: int, num_prompts: int,
                 *, emb_cpu=True):
        """initialize the model

        Args:
            dim_input: dimension of data input
            dim_prompt: dimension of prompt embedding
            dim_output: dimension of the output
            num_prompts: number of prompts
            emb_cpu: whether to store the prompt embedding on cpu
        """
        super().__init__()
        self.dim_input = dim_input
        self.dim_prompt = dim_prompt
        self.dim_output = dim_output
        self.emb_cpu = emb_cpu

        class WeightModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc_prompt = nn.Linear(dim_prompt, dim_output)
                self.fc = nn.Linear(dim_input, dim_output)

        self.weights = WeightModule()
        factory_kwargs = {'device': 'cpu'} if self.emb_cpu else {}
        self.prompt_embedding = nn.Embedding(num_prompts, dim_prompt, dtype=torch.float32, sparse=True,
                                             **factory_kwargs)

    def to(self, *args, **kwargs):
        """move the model to the specified device, excluding the embedding layer"""
        self.weights.to(*args, **kwargs)
        if not self.emb_cpu:
            self.prompt_embedding = self.prompt_embedding.to(*args, **kwargs)

    def forward(self, inputs, prompt):
        if not self.emb_cpu:
            prompt = prompt.to(self.prompt_embedding.weight.device)
        prompt_embedding = self.prompt_embedding(prompt)
        if self.emb_cpu:
            prompt_embedding = prompt_embedding.to(device=self.weights.fc_prompt.weight.device)
        prompt_output = self.weights.fc_prompt(prompt_embedding)

        data_output = self.weights.fc(inputs)
        return prompt_output + data_output
