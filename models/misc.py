import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Optional, Callable, Union, Tuple

class ResidualDAE(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        tied : bool,
        activation: Optional[Callable] = None,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension

        self.weights = nn.Parameter(
            torch.randn(input_dimension, hidden_dimension)
        )

        self.bias = nn.Parameter(
            torch.zeros(hidden_dimension)
        )
        if tied:
            self.weights_out = self.weights
        else:
            self.weights_out = nn.Parameter(
                torch.randn(input_dimension, hidden_dimension)
            )

        # softplus(beta) = 1 initially
        beta_init = math.log(math.expm1(1.0))
        self.beta = nn.Parameter(torch.tensor(beta_init))

        self.activation = activation() if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor, bias_factor : Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x @ self.weights / math.sqrt(self.input_dimension)
        if bias_factor is not None:
            h = h + bias_factor[:,None]*self.bias[None,:]
        else:
            h = h + self.bias
        b = F.softplus(self.beta)

        return b * x + self.activation(h) @ self.weights_out.T / math.sqrt(self.hidden_dimension)

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            (x.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2)

class Blend(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        gamma: float,
        learn_gamma: bool = False,
        interpolate: bool = False,
    ):
        super().__init__()

        self.module = module
        self.interpolate = interpolate

        if interpolate and not 0.0 <= gamma <= 1.0:
            raise ValueError(
                "For interpolation, gamma must be between 0 and 1."
            )

        if learn_gamma:
            if interpolate:
                eps = 1e-6
                gamma = min(max(gamma, eps), 1.0 - eps)

                self.raw_gamma = nn.Parameter(
                    torch.tensor(
                        math.log(gamma / (1.0 - gamma)),
                        dtype=torch.float32,
                    )
                )
            else:
                self.raw_gamma = nn.Parameter(
                    torch.tensor(
                        math.log(math.expm1(gamma)),
                        dtype=torch.float32,
                    )
                )

            self.gamma = None

        else:
            self.register_buffer(
                "gamma",
                torch.tensor(float(gamma)),
            )
            self.raw_gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.module(x)

        if y.shape != x.shape:
            raise ValueError(
                f"ResidualBlend requires matching shapes, "
                f"but got input {tuple(x.shape)} and output {tuple(y.shape)}."
            )

        if self.raw_gamma is not None:
            gamma = (
                torch.sigmoid(self.raw_gamma)
                if self.interpolate
                else F.softplus(self.raw_gamma)
            )
        else:
            gamma = self.gamma

        if self.interpolate:
            return (1.0 - gamma) * x + gamma * y

        return x + gamma * y

class TokenMasker(nn.Module):
    def __init__(self, mask_ratios: Union[float, Tuple[float, float]], return_visible: bool = False,
                 mask_dtype: torch.dtype = torch.float32):
        super(TokenMasker, self).__init__()
        self.return_visible = return_visible
        self.mask_dtype = mask_dtype
        self.update_mask_ratios(mask_ratios)
        
    def update_mask_ratios(self, mask_ratios: Union[float, Tuple[float, float]]):
        if isinstance(mask_ratios, float):
            self.unique_mask_ratio = True
            self.mask_ratio_min = float(mask_ratios)
            self.mask_ratio_max = float(mask_ratios)
        elif isinstance(mask_ratios, tuple) and len(mask_ratios) == 2:
            self.unique_mask_ratio = False
            self.mask_ratio_min = float(mask_ratios[0])
            self.mask_ratio_max = float(mask_ratios[1])
        else:
            raise ValueError("mask_ratios must be a float or a tuple of two floats")
    @staticmethod
    def mask_with_indices(x : torch.Tensor, visible_indices : torch.Tensor):
        """
        x: (B, N, ...)
        visible_indices: (B, K)
        
        Returns:
          x_visible: (B, K, ...)
        """
        B = x.shape[0]
        idx = visible_indices
        # expand idx to match trailing dims of x for gather
        while idx.dim() < x.dim():
            idx = idx.unsqueeze(-1)
        # we expand to (B, N_visible, ...)
        idx = idx.expand(*visible_indices.shape, *x.shape[2:])
        # finally gather
        x_visible = torch.gather(x, dim=1, index=idx)
        return x_visible
    
    @staticmethod
    def unmask_from_restore_indices(x_visible : torch.Tensor, restore_indices : torch.Tensor):
        """
        x_visible: (B, K, ...)
        restore_indices: (B, N)
        
        Returns:
          x: (B, N, ...)
        """
        B, N = restore_indices.shape
        K = x_visible.shape[1]
        # create empty tensor
        x_shape = (B, N) + x_visible.shape[2:]
        x = torch.zeros(x_shape, device=x_visible.device, dtype=x_visible.dtype)
        # expand restore_indices to match trailing dims of x_visible for scatter
        idx = restore_indices
        while idx.dim() < x_visible.dim():
            idx = idx.unsqueeze(-1)
        idx = idx.expand(*restore_indices.shape, *x_visible.shape[2:])
        # scatter
        x = x.scatter(dim=1, index=idx, src=x_visible)
        return x
    def forward(self, x: torch.Tensor):
        """
        x: (B, N, ...). Only N is used for masking; the rest are treated as token dims.

        Returns:
          visible_indices: (B, K)
          restore_indices: (B, N)
          mask: (B, N) with 0=visible, 1=masked (dtype = mask_dtype)
          (optional) x_visible: (B, K, ...)
        """
        device = x.device
        B, N = x.shape[:2]

        if self.unique_mask_ratio:
            mask_ratio = self.mask_ratio_min
        else:
            r = torch.rand(1, device=device).item()
            mask_ratio = r * (self.mask_ratio_max - self.mask_ratio_min) + self.mask_ratio_min

        # clamp for safety
        mask_ratio = float(max(0.0, min(1.0, mask_ratio)))

        num_visible = int(round((1.0 - mask_ratio) * N))
        num_visible = max(1, min(N, num_visible))  # keep in [1, N]

        # per-sample shuffle
        shuffled = torch.rand(B, N, device=device).argsort(dim=1)
        visible_indices = shuffled[:, :num_visible]
        restore_indices = shuffled.argsort(dim=1)

        # mask in shuffled order: first K are visible (0), rest masked (1)
        mask = torch.ones((B, N), device=device)
        mask[:, :num_visible] = 0
        # unshuffle to original token order
        mask = mask.gather(dim=1, index=restore_indices)
        mask = mask.to(dtype=self.mask_dtype)
        if self.return_visible:
            x_visible = self.mask_with_indices(x, visible_indices)
            return x_visible, visible_indices, restore_indices, mask
        else:
            return visible_indices, restore_indices, mask

class GRN(nn.Module):
    def __init__(self, channels: int, num_dims: int, eps: float = 1e-6):
        super().__init__()
        self.num_dims = num_dims
        self.gamma = nn.Parameter(torch.zeros(1, channels, *([1] * num_dims)))
        self.beta = nn.Parameter(torch.zeros(1, channels, *([1] * num_dims)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 + self.num_dims:
            raise ValueError(f"Input tensor must have {2 + self.num_dims} dimensions, but got {x.dim()}.")
        gx = torch.linalg.vector_norm(x, ord=2, dim=tuple(range(2, 2 + self.num_dims)), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta

class Patchifier(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_dims: int,
        pad: bool,
        in_channels_first: bool,
        out_channels_position: str = "first",
        padding_mode: str = "constant",
        padding_value: float = 0.0,
    ):
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")

        if num_dims <= 0:
            raise ValueError("num_dims must be positive.")

        if out_channels_position not in {"first", "middle", "last"}:
            raise ValueError(
                "out_channels_position must be one of "
                "'first', 'middle', or 'last', "
                f"but got '{out_channels_position}'."
            )

        self.patch_size = patch_size
        self.num_dims = num_dims
        self.pad = pad
        self.in_channels_first = in_channels_first
        self.out_channels_position = out_channels_position
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.num_dims + 2:
            raise ValueError(
                f"Input tensor must have {self.num_dims + 2} dimensions, "
                f"but got {x.ndim}."
            )

        # Convert internally to [B, C, spatial...]
        if self.in_channels_first:
            _, _, *dims = x.shape
        else:
            _, *dims, _ = x.shape
            x = x.movedim(-1, 1)

        # Pad spatial dimensions to multiples of patch_size
        if self.pad:
            padding = [(-dim) % self.patch_size for dim in dims]
            pad_spec = [v for p in reversed(padding) for v in (0, p)]

            if any(padding):
                if self.padding_mode == "constant":
                    x = F.pad(
                        x,
                        pad_spec,
                        mode=self.padding_mode,
                        value=self.padding_value,
                    )
                else:
                    x = F.pad(
                        x,
                        pad_spec,
                        mode=self.padding_mode,
                    )

        else:
            for dim in dims:
                if dim % self.patch_size != 0:
                    raise ValueError(
                        f"Dimension size {dim} is not divisible by "
                        f"patch size {self.patch_size} and padding is disabled."
                    )

        # [B, C, d1, ..., dn]
        #     ->
        # [B, C, n1, ..., nn, p1, ..., pn]
        for dim in range(2, 2 + self.num_dims):
            x = x.unfold(
                dimension=dim,
                size=self.patch_size,
                step=self.patch_size,
            )

        if self.out_channels_position == "middle":
            # [B, n1, ..., nn, C, p1, ..., pn]
            x = x.movedim(1, 1 + self.num_dims)

        elif self.out_channels_position == "last":
            # [B, n1, ..., nn, p1, ..., pn, C]
            x = x.movedim(1, -1)

        # "first":
        # [B, C, n1, ..., nn, p1, ..., pn]

        return x

class ConvPatchEmbedding1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[Callable] = None,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        if num_conv_layers < 1:
            raise ValueError("num_conv_layers must be >= 1")

        layers = []
        c_in = in_channels

        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv1d(
                    c_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )

            layers.append(activation() if activation is not None else nn.GELU())
            c_in = out_channels

        self.net = nn.Sequential(*layers)
        self.norm = norm_layer(out_channels)

        self.attn_pool = nn.Conv2d(out_channels, 1, kernel_size=1, bias=bias)

        nn.init.zeros_(self.attn_pool.weight)
        if bias:
            nn.init.zeros_(self.attn_pool.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected (B, N, C, L)")

        B, N, C, L = x.shape

        x = rearrange(x, "b n c l -> (b n) c l")
        x = self.net(x)

        w = torch.softmax(self.attn_pool(x), dim=-1)
        x = (x * w).sum(dim=-1)

        x = self.norm(x)

        return rearrange(x, "(b n) c -> b n c", b=B, n=N)

class ConvPatchEmbedding2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[Callable] = None,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        if num_conv_layers < 1:
            raise ValueError("num_conv_layers must be >= 1")

        layers = []
        c_in = in_channels

        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv2d(
                    c_in,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )

            layers.append(activation() if activation is not None else nn.GELU())
            c_in = out_channels

        self.net = nn.Sequential(*layers)
        self.norm = norm_layer(out_channels)

        self.attn_pool = nn.Conv2d(out_channels, 1, kernel_size=1, bias=bias)

        nn.init.zeros_(self.attn_pool.weight)
        if bias:
            nn.init.zeros_(self.attn_pool.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("Expected (B, N, C, H, W)")

        B, N, C, H, W = x.shape

        x = rearrange(x, "b n c h w -> (b n) c h w")
        x = self.net(x)

        w = self.attn_pool(x).flatten(2)   # (B*N, 1, H'*W')
        w = torch.softmax(w, dim=-1)

        x = x.flatten(2)                   # (B*N, C, H'*W')
        x = (x * w).sum(dim=-1)            # (B*N, C)

        x = self.norm(x)

        return rearrange(x, "(b n) c -> b n c", b=B, n=N)

class ConvPatchEmbedding3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_conv_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[Callable] = None,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        if num_conv_layers < 1:
            raise ValueError("num_conv_layers must be >= 1")

        layers = []
        c_in = in_channels

        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv3d(
                    in_channels=c_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                )
            )

            layers.append(
                activation() if activation is not None else nn.GELU()
            )

            c_in = out_channels

        self.net = nn.Sequential(*layers)
        self.norm = norm_layer(out_channels)

        self.attn_pool = nn.Conv3d(
            out_channels,
            1,
            kernel_size=1,
            bias=bias,
        )

        # Start as uniform average pooling
        nn.init.zeros_(self.attn_pool.weight)
        if bias:
            nn.init.zeros_(self.attn_pool.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError("Expected input shape (B, N, C, D, H, W)")

        B, N, C, D, H, W = x.shape

        x = rearrange(
            x,
            "b n c d h w -> (b n) c d h w",
        )

        x = self.net(x)

        # (B*N, 1, D', H', W') -> (B*N, 1, D'*H'*W')
        w = self.attn_pool(x).flatten(2)
        w = torch.softmax(w, dim=-1)

        # (B*N, C, D', H', W') -> (B*N, C, D'*H'*W')
        x = x.flatten(2)

        # Weighted spatial pooling
        x = (x * w).sum(dim=-1)

        x = self.norm(x)

        return rearrange(
            x,
            "(b n) c -> b n c",
            b=B,
            n=N,
        )