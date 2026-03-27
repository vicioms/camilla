import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, List, Optional
from psim_kernels import neigh_count, neigh_list
import math

def q_matrices_process_3d(coefs, basis):
    # coefs: (num_matrices, num_basis)
    # basis: (num_basis, 3, 3)
    q = torch.einsum('mb,bij->mij', coefs, basis)

    q_vals, q_vecs = torch.linalg.eigh(q)          # (m,3), (m,3,3)

    q_vals_exp = torch.exp(q_vals) 
    exp_q = torch.einsum('mi,mij,mj->mij',
                          q_vals_exp,
                          torch.eye(3).expand(q.shape[0],-1,-1).to(q.device),
                          q_vals_exp)
    exp_q = (q_vecs * q_vals_exp[:, None, :]) @ q_vecs.mT   # (m,3,3)

    dexp = q_vals_exp[:, :, None] - q_vals_exp[:, None, :]  # (m,3,3)
    dvals = q_vals[:, :, None]     - q_vals[:, None, :]      # (m,3,3)
    F = torch.where(
        dvals.abs() > 1e-10,
        dexp / dvals,
        q_vals_exp[:, :, None].expand_as(dvals)
    ) 

    basis_proj = torch.einsum('mki,bij,mjl->mbkl', q_vecs, basis, q_vecs)  # (m,b,3,3)

    F_times_proj = F[:, None, :, :] * basis_proj              # (m,b,3,3), Hadamard
    exp_gradients = torch.einsum('mik,mbkl,mjl->mbij',
                                  q_vecs, F_times_proj, q_vecs)  # (m,b,3,3)

    return exp_q, exp_gradients

class PSimSystem:
    particle_id : torch.Tensor
    pos : torch.Tensor
    force : torch.Tensor
    shape : torch.Tensor
    cell_id : torch.Tensor
    cell_offsets : torch.Tensor
    neigh_counts : torch.Tensor
    neigh_count_offsets : torch.Tensor
    neigh_list : torch.Tensor
    
    
    def __init__(self, num_cells_per_dim : Union[int, Tuple[int, int, int]], cell_size : Union[float, Tuple[float, float, float]], device='cuda'):
        if isinstance(num_cells_per_dim, Tuple):
            self.num_cells_per_dim = num_cells_per_dim
            self.num_cells = self.num_cells_per_dim[0] * self.num_cells_per_dim[1] * self.num_cells_per_dim[2]
        elif isinstance(num_cells_per_dim, int):
            self.num_cells_per_dim = (num_cells_per_dim, num_cells_per_dim, num_cells_per_dim)
            self.num_cells = num_cells_per_dim ** 3
        else:
            raise ValueError("num_cells_per_dim must be either an integer or a tuple of three integers")
        self.num_cells_per_dim_tensor = torch.tensor(self.num_cells_per_dim, dtype=torch.long, device=device)
        if isinstance(cell_size, Tuple):
            self.cell_size = cell_size
        elif isinstance(cell_size, float):
            self.cell_size = (cell_size, cell_size, cell_size)
        else:
            raise ValueError("cell_size must be either a float or a tuple of three floats")
        self.cell_size_vec = torch.tensor(self.cell_size, dtype=torch.float32, device=device)
        self.box_size = (self.num_cells_per_dim[0] * self.cell_size[0], 
                            self.num_cells_per_dim[1] * self.cell_size[1], 
                            self.num_cells_per_dim[2] * self.cell_size[2])
        self.box_size_vec = torch.tensor(self.box_size, dtype=torch.float32, device=device)
        self.cell_offsets = torch.empty(self.num_cells + 1, dtype=torch.long, device=device)
        self.device = device

    def update_neighbor_list(self):
        cell_coords = torch.floor(self.pos / self.cell_size_vec).long() % self.num_cells_per_dim_tensor
        nx, ny, nz = self.num_cells_per_dim
        cell_id = cell_coords[:, 0] + (cell_coords[:, 1] + cell_coords[:, 2] * ny)*nx
        cell_id_sorted, perm = torch.sort(cell_id) 
        self.cell_id = cell_id_sorted
        self.particle_id = self.particle_id[perm].contiguous()
        self.pos = self.pos[perm].contiguous()
        self.shape = self.shape[perm].contiguous()
        counts = torch.bincount(self.cell_id, minlength=self.num_cells)
        self.cell_offsets[0] = 0
        self.cell_offsets[1:] = torch.cumsum(counts, dim=0)
        neigh_count(pos=self.pos, cell_ids=self.cell_id, cell_offsets=self.cell_offsets, neigh_counts=self.neigh_counts)
        self.neigh_count_offsets[0] = 0
        self.neigh_count_offsets[1:] = torch.cumsum(self.neigh_counts, dim=0)
        total_neighbors = self.neigh_count_offsets[-1]
        if self.neigh_list is None or self.neigh_list.shape[0] < total_neighbors:
            self.neigh_list = torch.empty(total_neighbors, dtype=torch.long, device=self.device)    
        neigh_list(pos=self.pos, cell_ids=self.cell_id, cell_offsets=self.cell_offsets, neigh_counts=self.neigh_counts, neigh_count_offsets=self.neigh_count_offsets, neigh_list=self.neigh_list)
