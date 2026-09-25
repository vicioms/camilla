import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from einops import rearrange




class ConvHopfield3d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        num_memories: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        beta: float = 1.0,
        conv_bias: bool = False,
        softmax_bias: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_memories = num_memories
        self.beta = beta

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        if any(k % 2 == 0 for k in kernel_size):
            raise ValueError(
                "Kernel sizes must be odd to preserve spatial dimensions."
            )

        padding = tuple((k - 1) // 2 for k in kernel_size)

        self.query_conv = nn.Conv3d(
            in_channels=input_dim,
            out_channels=memory_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=conv_bias,
        )

        self.memories = nn.Linear(
            memory_dim,
            num_memories,
            bias=softmax_bias,
        )

        self.out_conv = nn.Conv3d(
            in_channels=memory_dim,
            out_channels=input_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected input shape (B, C, D, H, W)")

        B, C, D, H, W = x.shape

        if C != self.input_dim:
            raise ValueError(
                f"Input channels {C} do not match input_dim={self.input_dim}."
            )

        # Locally informed queries
        q = self.query_conv(x)
        q = rearrange(q, "b c d h w -> b (d h w) c")

        # Hopfield memory lookup
        logits = self.memories(q)
        weights = F.softmax(self.beta * logits, dim=-1)

        # Tied key/value memories
        output = weights @ self.memories.weight

        output = rearrange(
            output,
            "b (d h w) c -> b c d h w",
            d=D,
            h=H,
            w=W,
        )

        output = self.out_conv(output)

        return output 



