import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union, Tuple, Optional, Callable
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

# logic of RoPE
# x is (..., N, D)
#positions = torch.arange(x.shape[-2], device=x.device) (can be custom positions)
#omegas = 1.0 / (10000 ** (torch.arange(0, x.shape[-1], 2, device=x.device) / x.shape[-1]))
#pos_omegas = torch.einsum('n,d->nd', positions, omegas)
#cos = pos_omegas.cos()
#cos = torch.stack([cos, cos], dim=-1).flatten(-2)
#sin = pos_omegas.sin()
#sin = torch.stack([sin, sin], dim=-1).flatten(-2)
#def rotate_half(x):
#    return torch.stack([-x[...,1::2], x[...,::2]], dim=-1).reshape_as(x)

class TransformerLayerRoPE(nn.Module):
    def __init__(self, emb_dim : int,
                 nheads : int,
                 dim_feedforward : int,
                 dropout : float = 0.1,
                 layer_scale : Optional[float] = None,
                 bias : bool = True,
                 norm_layer : Callable = nn.LayerNorm):
        super(TransformerLayerRoPE, self).__init__()
        assert emb_dim % nheads == 0, "emb_dim must be divisible by nheads"
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.head_dim = emb_dim // nheads
        self.dim_feedforward = dim_feedforward
        self.dropout = nn.Dropout(dropout)
        self.pos_emb = RotaryEmbedding(dim=self.head_dim)

        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.norm1 = norm_layer(emb_dim)
        self.norm2 = norm_layer(emb_dim)

        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim, bias=bias),
        )

        if layer_scale is None:
            self.use_layer_scale = False
        else:
            if layer_scale > 0.0:
                self.use_layer_scale = True
                self.layer_scale_1 = nn.Parameter(layer_scale * torch.ones((emb_dim)), requires_grad=True)
                self.layer_scale_2 = nn.Parameter(layer_scale * torch.ones((emb_dim)), requires_grad=True)
            else:
                self.use_layer_scale = False
                

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, H, N, Dh)
        B, N, D = x.shape
        return x.view(B, N, self.nheads, self.head_dim).transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, N, Dh) -> (B, N, D)
        B, H, N, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Dh)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, positions: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            q = self.pos_emb.rotate_queries_or_keys(q)
            k = self.pos_emb.rotate_queries_or_keys(k)
            return q, k
        max_pos = int(positions.max().item())
        pos = torch.arange(max_pos + 1, device=positions.device)
        freqs = self.pos_emb(pos)
        freqs_batch = freqs[positions]  # (B, N, Dh)
        freqs_batch = freqs_batch.unsqueeze(1) # (B, 1, N, Dh)
        q = apply_rotary_emb(freqs_batch, q)
        k = apply_rotary_emb(freqs_batch, k)
        return q, k
    def forward(self, src: torch.Tensor,
                positions : Optional[torch.Tensor] = None,
                attn_mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm
        x = self.norm1(src)
        q = self._shape_heads(self.q_proj(x))  # (B, H, N, Dh)
        k = self._shape_heads(self.k_proj(x))  # (B, H, N, Dh)
        v = self._shape_heads(self.v_proj(x))  # (B, H, N, Dh)
        q, k = self._apply_rope(q, k, positions)  # apply RoPE
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, N, Dh)
        attn_out = self._merge_heads(attn_out) # (B, N, D)
        attn_out = self.out_proj(attn_out)

        if self.use_layer_scale:
            src = src + self.dropout(attn_out)*self.layer_scale_1[None, None, :]
        else:
            src = src + self.dropout(attn_out)

        y = self.ffn(self.norm2(src))

        if self.use_layer_scale:
            src = src + self.dropout(y)*self.layer_scale_2[None, None, :]
        else:
            src = src + self.dropout(y)
        return src