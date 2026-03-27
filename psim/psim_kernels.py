from networkx import sigma
import torch
import triton
import triton.language as tl




@triton.jit
def _neigh_count(
    pos_ptr,
    cell_ids_ptr,
    cell_offsets_ptr,
    num_particles,
    nx, ny, nz,
    Lx, Ly, Lz,
    cutoff_skin2,
    neigh_counts_ptr,
    RING: tl.constexpr,
    USE_PBC: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_particles:
        return

    x_i = tl.load(pos_ptr + pid * 3 + 0)
    y_i = tl.load(pos_ptr + pid * 3 + 1)
    z_i = tl.load(pos_ptr + pid * 3 + 2)
    cid_i = tl.load(cell_ids_ptr + pid)
    cx_i = cid_i % nx
    cy_i = (cid_i // nx) % ny
    cz_i = cid_i // (nx * ny)

    neigh_count = 0

    for ddz in range(-RING, RING + 1):
        for ddy in range(-RING, RING + 1):
            for ddx in range(-RING, RING + 1):

                cx_j = cx_i + ddx
                cy_j = cy_i + ddy
                cz_j = cz_i + ddz

                if USE_PBC:
                    cx_j = cx_j % nx
                    cy_j = cy_j % ny
                    cz_j = cz_j % nz
                    valid_cell = True
                else:
                    valid_cell = (
                        (cx_j >= 0) * (cx_j < nx) *
                        (cy_j >= 0) * (cy_j < ny) *
                        (cz_j >= 0) * (cz_j < nz)
                    )
                    cx_j = tl.where(valid_cell, cx_j, 0)
                    cy_j = tl.where(valid_cell, cy_j, 0)
                    cz_j = tl.where(valid_cell, cz_j, 0)

                cid_j = cx_j + nx * (cy_j + ny * cz_j)
                start = tl.load(cell_offsets_ptr + cid_j)
                end   = tl.load(cell_offsets_ptr + cid_j + 1)

                if not USE_PBC:
                    start = tl.where(valid_cell, start, 0)
                    end   = tl.where(valid_cell, end,   0)

                j = start
                while j < end:
                    offs_j = j + tl.arange(0, BLOCK_J)
                    mask_j = offs_j < end

                    x_j = tl.load(pos_ptr + offs_j * 3 + 0, mask=mask_j, other=0.0)
                    y_j = tl.load(pos_ptr + offs_j * 3 + 1, mask=mask_j, other=0.0)
                    z_j = tl.load(pos_ptr + offs_j * 3 + 2, mask=mask_j, other=0.0)

                    dx = x_i - x_j
                    dy = y_i - y_j
                    dz = z_i - z_j

                    if USE_PBC:
                        dx = dx - Lx * tl.round(dx / Lx)
                        dy = dy - Ly * tl.round(dy / Ly)
                        dz = dz - Lz * tl.round(dz / Lz)

                    r2 = dx*dx + dy*dy + dz*dz

                    valid = (
                        mask_j &
                        (offs_j != pid) &
                        (r2 < cutoff_skin2)
                    )

                    neigh_count += tl.sum(valid.to(tl.int32), axis=0)
                    j += BLOCK_J

    tl.store(neigh_counts_ptr + pid, neigh_count)

@triton.jit
def _neigh_list(
    pos_ptr,
    cell_ids_ptr,
    cell_offsets_ptr,
    num_particles,
    nx, ny, nz,
    Lx, Ly, Lz,
    cutoff_skin2,
    neigh_count_offsets_ptr,
    neigh_list_ptr,
    RING: tl.constexpr,
    USE_PBC: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_particles:
        return

    x_i = tl.load(pos_ptr + pid * 3 + 0)
    y_i = tl.load(pos_ptr + pid * 3 + 1)
    z_i = tl.load(pos_ptr + pid * 3 + 2)
    cid_i = tl.load(cell_ids_ptr + pid)
    cx_i = cid_i % nx
    cy_i = (cid_i // nx) % ny
    cz_i = cid_i // (nx * ny)

    list_start = tl.load(neigh_count_offsets_ptr + pid)
    list_end   = tl.load(neigh_count_offsets_ptr + pid + 1)
    list_idx   = list_start

    for ddz in range(-RING, RING + 1):
        for ddy in range(-RING, RING + 1):
            for ddx in range(-RING, RING + 1):
                cx_j = cx_i + ddx
                cy_j = cy_i + ddy
                cz_j = cz_i + ddz

                if USE_PBC:
                    cx_j = cx_j % nx
                    cy_j = cy_j % ny
                    cz_j = cz_j % nz
                    valid_cell = 1
                else:
                    valid_cell = (
                        (cx_j >= 0) * (cx_j < nx) *
                        (cy_j >= 0) * (cy_j < ny) *
                        (cz_j >= 0) * (cz_j < nz)
                    )

                cx_j = tl.where(valid_cell, cx_j, 0)
                cy_j = tl.where(valid_cell, cy_j, 0)
                cz_j = tl.where(valid_cell, cz_j, 0)
                cid_j = cx_j + nx * (cy_j + ny * cz_j)

                start = tl.load(cell_offsets_ptr + cid_j)
                end   = tl.load(cell_offsets_ptr + cid_j + 1)
                if not USE_PBC:
                    start = tl.where(valid_cell, start, 0)
                    end   = tl.where(valid_cell, end,   0)

                j = start
                while j < end:
                    offs_j = j + tl.arange(0, BLOCK_J)
                    mask_j = offs_j < end

                    x_j = tl.load(pos_ptr + offs_j * 3 + 0, mask=mask_j, other=0.0)
                    y_j = tl.load(pos_ptr + offs_j * 3 + 1, mask=mask_j, other=0.0)
                    z_j = tl.load(pos_ptr + offs_j * 3 + 2, mask=mask_j, other=0.0)

                    dx = x_i - x_j
                    dy = y_i - y_j
                    dz = z_i - z_j
                    if USE_PBC:
                        dx = dx - Lx * tl.round(dx / Lx)
                        dy = dy - Ly * tl.round(dy / Ly)
                        dz = dz - Lz * tl.round(dz / Lz)

                    r2 = dx*dx + dy*dy + dz*dz
                    valid = (
                        mask_j &
                        (offs_j != pid) &
                        (r2 < cutoff_skin2)
                    )

                    tl.store(
                        neigh_list_ptr + list_idx + tl.arange(0, BLOCK_J),
                        offs_j,
                        mask=valid,
                    )
                    list_idx += tl.sum(valid.to(tl.int32), axis=0)
                    j += BLOCK_J
    

def neigh_count(
    pos,
    cell_ids,
    cell_offsets,
    num_cells_per_dim,
    box_size,
    cutoff_skin,
    neigh_count,
    neigh_count_offsets,
    neigh_list,
    ring=1,
    use_pbc=True,
    block_j=64
):
    assert pos.is_contiguous()
    assert cell_ids.is_contiguous()
    assert cell_offsets.is_contiguous()
    assert neigh_count_offsets.is_contiguous()
    assert neigh_list.is_contiguous()

    num_particles = pos.shape[0]
    nx, ny, nz = num_cells_per_dim
    Lx, Ly, Lz = box_size
    cutoff_skin2 = cutoff_skin * cutoff_skin

    # First, count the number of neighbors for each particle
    _neigh_count[(num_particles,)](
        pos_ptr=pos.view(-1),
        cell_ids_ptr=cell_ids,
        cell_offsets_ptr=cell_offsets,
        num_particles=num_particles,
        nx=nx, ny=ny, nz=nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        cutoff_skin2=cutoff_skin2,
        neigh_counts_ptr=neigh_count,
        RING=ring,
        USE_PBC=use_pbc,
        BLOCK_J=block_j
    )
    
    
def neigh_list(
    pos,
    cell_ids,
    cell_offsets,
    num_cells_per_dim,
    box_size,
    cutoff_skin,
    neigh_count_offsets,
    neigh_list,
    ring=1,
    use_pbc=True,
    block_j=64
):
    assert pos.is_contiguous()
    assert cell_ids.is_contiguous()
    assert cell_offsets.is_contiguous()
    assert neigh_count_offsets.is_contiguous()
    assert neigh_list.is_contiguous()

    num_particles = pos.shape[0]
    nx, ny, nz = num_cells_per_dim
    Lx, Ly, Lz = box_size
    cutoff_skin2 = cutoff_skin * cutoff_skin

    # Then, populate the neighbor list
    _neigh_list[(num_particles,)](
        pos_ptr=pos.view(-1),
        cell_ids_ptr=cell_ids,
        cell_offsets_ptr=cell_offsets,
        num_particles=num_particles,
        nx=nx, ny=ny, nz=nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        cutoff_skin2=cutoff_skin2,
        neigh_count_offsets_ptr=neigh_count_offsets,
        neigh_list_ptr=neigh_list,
        RING=ring,
        USE_PBC=use_pbc,
        BLOCK_J=block_j
    )





























# TLJ:
# V(r) = 4 * epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
# F(r) = - V'(r) = 24 * epsilon * [ 2*(sigma^12/r^13) - (sigma^6/r^7) ] 
# F(r_c) = 24 * epsilon * [ 2*(sigma^12/r_c^13) - (sigma^6/r_c^7) ]
# F_trunc(r) = F(r) - F(r_c)

# Morse:
# V(r) = V_eq * (1 - exp(-a*(r-r_eq)))^2
# V'(r) = 2 * a * V_eq * (1 - exp(-a*(r-r_eq))) * exp(-a*(r-r_eq))
# F(r) = - V'(r) = - 2 * a * V_eq * (1 - exp(-a*(r-r_eq))) * exp(-a*(r-r_eq))
# F(r_c) = - 2 * a * V_eq * (1 - exp(-a*(r_c-r_eq))) * exp(-a*(r_c-r_eq))
# F_trunc(r) = F(r) - F(r_c)
@triton.jit
def _pairwise_kernel(pos_ptr,
                     force_ptr,
                     cell_ids_ptr,
                     cell_offsets_ptr,
                     num_particles,
                     nx, ny, nz,
                     Lx, Ly, Lz,
                     cutoff2,
                     force_params,
                     RING: tl.constexpr,
                     POTENTIAL: tl.constexpr,
                     USE_PBC: tl.constexpr,
                     BLOCK_J: tl.constexpr):

    pid = tl.program_id(0)
    if pid >= num_particles:
        return

    base_i = 3 * pid
    x_i = tl.load(pos_ptr + base_i + 0)
    y_i = tl.load(pos_ptr + base_i + 1)
    z_i = tl.load(pos_ptr + base_i + 2)

    cid_i = tl.load(cell_ids_ptr + pid)

    n_xy = nx * ny
    cz_i = cid_i // n_xy
    rem = cid_i - cz_i * n_xy
    cy_i = rem // nx
    cx_i = rem - cy_i * nx

    fx = tl.full((), 0.0, tl.float32)
    fy = tl.full((), 0.0, tl.float32)
    fz = tl.full((), 0.0, tl.float32)

    for ddz in range(-RING, RING+1):
        for ddy in range(-RING, RING+1):
            for ddx in range(-RING, RING+1):
                cx_j = cx_i + ddx
                cy_j = cy_i + ddy
                cz_j = cz_i + ddz
                if USE_PBC:
                    cx_j = tl.where(cx_j < 0, cx_j + nx, cx_j)
                    cx_j = tl.where(cx_j >= nx, cx_j - nx, cx_j)
                    cy_j = tl.where(cy_j < 0, cy_j + ny, cy_j)
                    cy_j = tl.where(cy_j >= ny, cy_j - ny, cy_j)
                    cz_j = tl.where(cz_j < 0, cz_j + nz, cz_j)
                    cz_j = tl.where(cz_j >= nz, cz_j - nz, cz_j)
                    valid_cell = True
                else:
                    valid_cell = (
                    (cx_j >= 0) & (cx_j < nx) &
                    (cy_j >= 0) & (cy_j < ny) &
                    (cz_j >= 0) & (cz_j < nz)
                    )

                cid_j = cx_j + nx * (cy_j + ny * cz_j)
                cid_j = tl.where(valid_cell, cid_j, 0)

                start = tl.load(cell_offsets_ptr + cid_j)
                end = tl.load(cell_offsets_ptr + cid_j + 1)

                start = tl.where(valid_cell, start, 0)
                end = tl.where(valid_cell, end, start) # if not valid, set end to start to avoid processing invalid cells

                j = start
                while j < end:
                    offs_j = j + tl.arange(0, BLOCK_J)
                    mask_j = offs_j < end

                    base_j = 3 * offs_j
                    x_j = tl.load(pos_ptr + base_j + 0, mask=mask_j, other=0.0)
                    y_j = tl.load(pos_ptr + base_j + 1, mask=mask_j, other=0.0)
                    z_j = tl.load(pos_ptr + base_j + 2, mask=mask_j, other=0.0)

                    dx = x_i - x_j
                    dy = y_i - y_j
                    dz = z_i - z_j

                    if USE_PBC:
                        dx = dx - Lx * tl.floor(dx / Lx + 0.5)
                        dy = dy - Ly * tl.floor(dy / Ly + 0.5)
                        dz = dz - Lz * tl.floor(dz / Lz + 0.5)

                    r2 = dx * dx + dy * dy + dz * dz

                    valid = mask_j & (offs_j != pid) & (r2 < cutoff2) & (r2 > 0.0)

                    

                    if POTENTIAL == 0:  # force-shifted Lennard-Jones
                        intensity = tl.load(force_params)      # epsilon
                        scale = tl.load(force_params + 1)      # sigma

                        safe_r2 = tl.where(valid, r2, 1.0)
                        inv_r2  = 1.0 / safe_r2
                        rinv    = tl.sqrt(inv_r2)

                        s2  = (scale * scale) * inv_r2
                        s6  = s2 * s2 * s2

                        # F_vec = coeff * dr,  dr = r_i - r_j
                        coeff = 24.0 * intensity * (2.0 * s6 * s6 - s6) * inv_r2

                        # force-shift correction at cutoff
                        inv_rc2 = 1.0 / cutoff2
                        src2    = (scale * scale) * inv_rc2
                        src6    = src2 * src2 * src2
                        Frc     = 24.0 * intensity * (2.0 * src6 * src6 - src6) * tl.sqrt(inv_rc2)

                        coeff_fs = tl.where(valid, coeff - Frc * rinv, 0.0)

                        fx += tl.sum(coeff_fs * dx, axis=0)
                        fy += tl.sum(coeff_fs * dy, axis=0)
                        fz += tl.sum(coeff_fs * dz, axis=0)
                    elif POTENTIAL == 2:  # force-shifted Morse
                        potential_prefactor = tl.load(force_params)
                        location            = tl.load(force_params + 1)
                        inverse_scale       = tl.load(force_params + 2)

                        safe_r2 = tl.where(valid, r2, 1.0)
                        r       = tl.sqrt(safe_r2)
                        rinv    = tl.sqrt(1.0 / safe_r2)

                        exp_term       = tl.exp(-inverse_scale * (r - location))   # exp(-alpha*(r - r_eq))
                        one_m_exp_term  = 1.0 - exp_term                            # (1 - exp(-alpha*(r - r_eq)))

                        # dU/dr = 2 * V * alpha * e * (1 - e)
                        # F_vec = -dU/dr * r_hat = coeff * dr
                        coeff   = -2.0 * potential_prefactor * inverse_scale * exp_term * one_m_exp_term * rinv * rinv

                        # force-shift at cutoff
                        rc      = tl.sqrt(cutoff2)
                        exp_term_rc      = tl.exp(-inverse_scale * (rc - location))
                        one_m_exp_term_rc = 1.0 - exp_term_rc
                        Frc     = 2.0 * potential_prefactor * inverse_scale * exp_term_rc * one_m_exp_term_rc

                        coeff_fs = tl.where(valid, coeff + Frc * rinv, 0.0)

                        fx += tl.sum(coeff_fs * dx, axis=0)
                        fy += tl.sum(coeff_fs * dy, axis=0)
                        fz += tl.sum(coeff_fs * dz, axis=0)

                    else:
                        continue
                    j += BLOCK_J

    tl.store(force_ptr + base_i + 0, fx)
    tl.store(force_ptr + base_i + 1, fy)
    tl.store(force_ptr + base_i + 2, fz)
    



def pairwise_kernel_tlj(pos,
                        force,
                        cell_ids,
                        cell_offsets,
                        num_cells_per_dim,
                        box_size,
                        cutoff,
                        epsilon,
                        sigma,
                        ring=1,
                        block_j=64):
    assert pos.is_contiguous()
    assert force.is_contiguous()
    assert cell_offsets.is_contiguous()
    assert cell_ids.is_contiguous()

    num_particles = pos.shape[0]
    nx, ny, nz = num_cells_per_dim
    Lx, Ly, Lz = box_size
    cutoff2 = cutoff * cutoff
    force_params = torch.tensor([epsilon, sigma], dtype=torch.float32, device=pos.device)

    _pairwise_kernel[(num_particles,)](
        pos_ptr=pos.view(-1),
        force_ptr=force.view(-1),
        cell_offsets_ptr=cell_offsets,
        cell_ids_ptr=cell_ids,
        num_particles=num_particles,
        nx=nx, ny=ny, nz=nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        cutoff2=cutoff2,
        force_params=force_params,
        RING=ring,
        POTENTIAL=0,
        USE_PBC=1,
        BLOCK_J=block_j
    )


def pairwise_kernel_tmorse(pos,
                        force,
                        cell_ids,
                        cell_offsets,
                        num_cells_per_dim,
                        box_size,
                        cutoff,
                        intensity,
                        location,
                        inverse_scale,
                        ring=1,
                        block_j=64):
    assert pos.is_contiguous()
    assert force.is_contiguous()
    assert cell_offsets.is_contiguous()
    assert cell_ids.is_contiguous()

    num_particles = pos.shape[0]
    nx, ny, nz = num_cells_per_dim
    Lx, Ly, Lz = box_size
    cutoff2 = cutoff * cutoff
    force_params = torch.tensor([intensity, location, inverse_scale], dtype=torch.float32, device=pos.device)

    _pairwise_kernel[(num_particles,)](
        pos_ptr=pos.view(-1),
        force_ptr=force.view(-1),
        cell_offsets_ptr=cell_offsets,
        cell_ids_ptr=cell_ids,
        num_particles=num_particles,
        nx=nx, ny=ny, nz=nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        cutoff2=cutoff2,
        force_params=force_params,
        RING=ring,
        POTENTIAL=2,
        USE_PBC=1,
        BLOCK_J=block_j
    )


