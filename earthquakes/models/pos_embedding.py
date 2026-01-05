import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.offset_cached = None

    def forward(self, x, seq_dim=-2, offset=0):
        seq_len = x.size(seq_dim)
        need = (
            self.cos_cached is None
            or self.seq_len_cached != seq_len
            or self.offset_cached != offset
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        )
        if need:
            self.seq_len_cached = seq_len
            self.offset_cached = offset
            positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = positions[:, None] * self.inv_freq[None, :]  # [S, D//2]
            self.cos_cached = freqs.cos().type_as(x)
            self.sin_cached = freqs.sin().type_as(x)
        return self.cos_cached, self.sin_cached

def apply_rotary_pos_emb(x, cos, sin):
    # x: [batch_size, num_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim//2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)
