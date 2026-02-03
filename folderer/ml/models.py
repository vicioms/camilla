import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union, Tuple, Optional, Callable
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


class DropPath(nn.Module):
    """Stochastic Depth / DropPath for tensors of shape (B, ..., C)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # broadcast across all non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        binary_mask = random_tensor.floor()
        return x * binary_mask / keep_prob


class AdaptiveISAB(nn.Module):
    """
    Adaptive ISAB:
      H = LN( S + alpha1 * DropPath( MHA(S, X, X) ) )
      Y = LN( X + alpha2 * DropPath( MHA(X, H, H) ) )

    - S: learned slot queries (num_slots, dim)
    - alpha1/alpha2: LayerScale (trainable only if alpha_init > 0, otherwise fixed 1 via buffers)
    - DropPath applied on residual branches
    """
    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_heads: int,
        alpha_init: Optional[float] = 1e-3,
        drop_path_prob: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        # learned slot queries
        self.S = nn.Parameter(torch.randn(num_slots, dim))

        # attention
        self.attn1 = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)

        # norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # LayerScale: trainable only when alpha_init is provided and > 0
        if alpha_init is not None and alpha_init > 0.0:
            self.alpha1 = nn.Parameter(torch.full((dim,), float(alpha_init)))
            self.alpha2 = nn.Parameter(torch.full((dim,), float(alpha_init)))
        else:
            # fixed coefficient 1.0, NOT trainable
            self.register_buffer("alpha1", torch.ones(dim), persistent=False)
            self.register_buffer("alpha2", torch.ones(dim), persistent=False)

        # DropPath on each residual branch (separate modules for clarity)
        self.drop_path1 = DropPath(drop_path_prob)
        self.drop_path2 = DropPath(drop_path_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, dim)
        returns: (B, N, dim)
        """
        B = X.shape[0]
        S = self.S.unsqueeze(0).expand(B, -1, -1)  # (B, M, dim)

        H, _ = self.attn1(S, X, X)                 # (B, M, dim)
        H = self.ln1(S + self.alpha1 * self.drop_path1(H))

        Y, _ = self.attn2(X, H, H)                 # (B, N, dim)
        Y = self.ln2(X + self.alpha2 * self.drop_path2(Y))

        return Y
    

class TransformerLayer(nn.Module):
    def __init__(self, emb_dim : int,
                 nheads : int,
                 dim_feedforward : int,
                 dropout : float = 0.1,
                 layer_scale : Optional[float] = None,
                 bias : bool = True,
                 norm_layer : Callable = nn.LayerNorm):
        super(TransformerLayer, self).__init__()
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

    def forward(self, src: torch.Tensor,
                attn_mask : Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm
        x = self.norm1(src)
        q = self._shape_heads(self.q_proj(x))  # (B, H, N, Dh)
        k = self._shape_heads(self.k_proj(x))  # (B, H, N, Dh)
        v = self._shape_heads(self.v_proj(x))  # (B, H, N, Dh)
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