import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pos_embedding import RotaryPositionalEmbedding, apply_rotary_pos_emb

class AttentionBlock(nn.Module):
    def __init__(self, 
                 embed_dim : int, 
                 num_heads : int, 
                 dropout : float,
                 bias : bool,
                 rotary_pos_embed : bool) :
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.rotary_pos_embed = rotary_pos_embed
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if self.rotary_pos_embed:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
    
    def _reshape_to_heads(self, x : torch.Tensor) -> torch.Tensor :
        # in: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = x.size()
        # reshape as: [batch_size, seq_len, num_heads, head_dim]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # transpose to: [batch_size, num_heads, seq_len, head_dim]
        return x.transpose(1, 2).contiguous()
    
    def forward(self,
                x : torch.Tensor,
                attention_mask : Optional[torch.Tensor] = None,
                output_attentions : bool = False) :
        if output_attentions:
            raise NotImplementedError("output_attentions is not implemented yet.")
        batch_size, seq_len, _ = x.size()
        # dimensions become:
        # [batch_size, num_heads, seq_len, head_dim]
        query_x = self._reshape_to_heads(self.q_proj(x))
        key_x = self._reshape_to_heads(self.k_proj(x))
        value_x = self._reshape_to_heads(self.v_proj(x))

        if self.rotary_pos_embed:
            cos, sin = self.rotary_emb(query_x, seq_dim=2)
            query_x = apply_rotary_pos_emb(query_x, cos, sin)
            key_x = apply_rotary_pos_emb(key_x, cos, sin)


        # output here is [batch_size, num_heads, seq_len, head_dim]
        attn_output = F.scaled_dot_product_attention(
            query_x,
            key_x,
            value_x,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # switch num_heads and seq_len
        attn_output = attn_output.transpose(1, 2)
        # merge heads
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output
    

