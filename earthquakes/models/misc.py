import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Callable, Union, Tuple

class Patchifier1d(nn.Module):
    def __init__(self, patch_size: int, rearrange : bool = True, merge_channels_with_patches: bool = False, padding_mode: str = 'constant', padding_value: float = 0.0):
        super(Patchifier1d, self).__init__()
        self.patch_size = patch_size
        self.rearrange = rearrange
        self.merge_channels_with_patches = merge_channels_with_patches
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        if L % self.patch_size != 0:
            pad_length = self.patch_size - (L % self.patch_size)
            x = nn.functional.pad(x, (0, pad_length), mode=self.padding_mode, value=self.padding_value)
            L += pad_length
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        if(self.rearrange):
            if(self.merge_channels_with_patches):
                x = rearrange(x, 'b c l p -> b l (c p)')
                return x
            else:
                x = rearrange(x, 'b c l p -> b l c p')
                return x
        else:
            return x
class ConvPatchEncoder1d(nn.Module):
    def __init__(self, in_channels : int,
                 out_channels : int,
                 num_conv_layers : int, 
                 patch_size : int,
                 kernel_size : int = 3,
                 stride : int = 1,
                 padding : int = 0,
                 dilation : int = 1,
                 groups : int = 1,
                 bias : bool = True,
                 padding_mode : str = 'zeros',
                 activation : Optional[Callable] = None,
                 norm_layer : Callable = nn.LayerNorm):
        super(ConvPatchEncoder1d, self).__init__()
        self.patch_size = patch_size
        layers = []
        c_in = in_channels
        for i in range(num_conv_layers):
            conv = nn.Conv1d(in_channels=c_in,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=bias,
                             padding_mode=padding_mode)
            layers.append(conv)
            if activation is not None:
                layers.append(activation())
            else:
                layers.append(nn.GELU())
            c_in = out_channels
        
        self.net = nn.Sequential(*layers)
        self.norm = norm_layer(out_channels)
        #self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn_pool = nn.Conv1d(out_channels, 1, kernel_size=1, bias = bias)
        nn.init.zeros_(self.attn_pool.weight)
        if bias:
            nn.init.zeros_(self.attn_pool.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("Input tensor must be 4D (B, N, C,  L)")
        B, N, C, L = x.shape
        x = rearrange(x, 'b n c l -> (b n) c l')
        x = self.net(x)
        w = self.attn_pool(x) # (B*N, 1, L')
        w = torch.softmax(w, dim=-1)
        x = (x * w).sum(dim=-1) # (B*N, D)
        
        x = self.norm(x)
        x = rearrange(x, '(b n) c -> b n c', b=B, n=N)
        return x

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

