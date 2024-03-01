from typing import Type, Iterable
from collections import Counter

import torch
import torch.nn as nn

class NetworkCheck:
    def __init__(self, network: Type[nn.Module], n_channels: int, input_dims: Iterable[int]) -> None:
        self.network = network
        self.batch_size = 1
        self.n_channels = n_channels
        self.input_dims = list(input_dims)
        self.input_shape, self.output_shape = self.get_output_dimensions()
        
    def __str__(self) -> str:
        return f"Input dimensions: {self.input_shape}. Output dimensions: {self.output_shape}. Network runs successfully!"

    def get_output_dimensions(self) -> None:
        # input_sample: torch.Tensor = torch.randn(self.network.config.num_classes, *self.input_dims)
        input_sample: torch.Tensor = torch.randn(self.batch_size, self.n_channels, *self.input_dims)
        output: torch.Tensor = self.network(input_sample)
        return list(input_sample.shape), list(output.shape)
