from typing import Type, Iterable
from collections import Counter

import torch
import torch.nn as nn

class NetworkCheck:
    def __init__(self, network: Type[nn.Module], input_dims: Iterable[int]) -> None:
        self.network = network
        self.input_dims = list(input_dims)
        self.output_dims = self.get_output_dimensions()
        
    def __str__(self) -> str:
        return f"Input dimensions: {self.input_dims}. Output dimensions: {self.output_dims}. Network runs successfully!"

    def get_output_dimensions(self) -> None:
        input_sample: torch.Tensor = torch.randn(self.network.config.num_classes, *self.input_dims)
        output: torch.Tensor = self.network(input_sample)
        return list(output.shape)
