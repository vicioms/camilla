import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearST(nn.Module):
    def __init__(self, dim, bias):
        super(LinearST, self).__init__()
        self.linear = nn.Linear(dim, dim, bias=bias)
    def forward(self, x):
        v = self.linear(x)
        v_dot_x = (v * x).sum(dim=-1, keepdim=True)
        return v - v_dot_x * x
    

class SphericalPivoter(nn.Module):
    def __init__(self, dim, num_pivots, bias, use_spherical_projection):
        super(SphericalPivoter, self).__init__()
        if use_spherical_projection:
            self.q_proj = LinearST(dim, bias)
            self.k_proj = LinearST(dim, bias)
            self.v_proj = LinearST(dim, bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=bias)
            self.k_proj = nn.Linear(dim, dim, bias=bias)
            self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.num_pivots = num_pivots
        self.pivot_queries = nn.Parameter(F.normalize(torch.randn(num_pivots, dim), dim=-1))
        
    def forward(self, x):
        q = self.q_proj(self.pivot_queries)
        k = self.k_proj(x)
        v = self.v_proj(x)
        

class PivotSphericalAttention(nn.Module):
    def __init__(self, dim, num_pivots, bias, 
                 use_spherical_projection,

                 ):
        super(PivotSphericalAttention, self).__init__()
        if use_spherical_projection:
            self.q_proj = LinearST(dim, bias)
            self.k_proj = LinearST(dim, bias)
            self.v_proj = LinearST(dim, bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=bias)
            self.k_proj = nn.Linear(dim, dim, bias=bias)
            self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.num_pivots = num_pivots
        self.pivot_keys = nn.Parameter(F.normalize(torch.randn(num_pivots, dim), dim=-1))
        
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Compute attention weights using pivot tokens  
        