#include "cuda_runtime.h"
#include "curand.h"
#include "cell_list.cuh"
using namespace std;

void init_random_lattice_host(particle* particles, int num_particles, float3 cell_size, int seed)
{
    default_random_engine rng(seed);
    int n_lattice = int(ceil(pow(num_particles, 1.0f / 3.0f)));
    int num_grid_points = n_lattice * n_lattice * n_lattice;
    assert(num_particles <= num_grid_points);
    int* grid_indices = new int[num_grid_points];
    for (int i = 0; i < num_grid_points; i++)
        grid_indices[i] = i;
    std::shuffle(grid_indices, grid_indices + num_grid_points, rng);
    uniform_real_distribution<float> jitter(-0.4f, 0.4f);  // fraction of cell size
    for (int i = 0; i < num_particles; i++)
    {
        int grid_index = grid_indices[i];
        int z = grid_index / (n_lattice * n_lattice);
        int y = (grid_index - z * n_lattice * n_lattice) / n_lattice;
        int x = grid_index % n_lattice;
        particles[i].position = make_float4(
            (x + 0.5f + jitter(rng)) * cell_size.x,
            (y + 0.5f + jitter(rng)) * cell_size.y,
            (z + 0.5f + jitter(rng)) * cell_size.z,
            0.0f);
        particles[i].force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        particles[i].polarity = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    delete[] grid_indices;
}
int main()
{
    int num_particles = 80;
    int num_sites_per_side = 5;
    float spacing = 1.0f;
    float3 cell_size = make_float3(spacing, spacing, spacing);
    float3 box_size = make_float3(num_sites_per_side * spacing, num_sites_per_side * spacing, num_sites_per_side * spacing);
    int seed = 42;
    particle* host_particles = new particle[num_particles];
    init_random_lattice_host(host_particles, num_particles, cell_size, seed);
    cell_list clist(num_particles,
         cell_size, 
         box_size);
    cudaMemcpy(clist.particles, host_particles, num_particles * sizeof(particle), cudaMemcpyHostToDevice);
    delete[] host_particles;
    clist.update_list_pbc(256);
    cell_list host_clist(num_particles,
         cell_size, 
         make_float3(num_sites_per_side * spacing, num_sites_per_side * spacing, num_sites_per_side * spacing));
    clist.to_host(host_clist);
    for(int i = 0; i < num_particles; i++)
    {
        printf("Particle %d: position=(%f, %f, %f)\n", i, host_clist.particles[i].position.x, host_clist.particles[i].position.y, host_clist.particles[i].position.z);
    }
    for(int c_idx = 0; c_idx < host_clist.h_num_unique_cells; c_idx++)
    {
        int cell_id = host_clist.cell_ids[c_idx];
        int count = host_clist.cell_counts[c_idx];
        int offset = host_clist.cell_offsets[c_idx];
        int cell_start = host_clist.cell_start[cell_id];
        int cell_end = host_clist.cell_end[cell_id];
        printf("Cell %d: count=%d, offset=%d, start=%d, end=%d\n", cell_id, count, offset, cell_start, cell_end);
        int3 cell = unpack_cell_pbc_3(cell_id, host_clist.grid_size);
        printf("  Cell index: (%d, %d, %d)\n", cell.x, cell.y, cell.z);
        for(int dz = -1; dz <= 1; dz++)
        {
            for(int dy = -1; dy <= 1; dy++)
            {
                for(int dx = -1; dx <= 1; dx++)
                {
                    int3 neighbor_cell = make_int3(pmod(cell.x + dx, host_clist.grid_size.x),
                                                  pmod(cell.y + dy, host_clist.grid_size.y),
                                                  pmod(cell.z + dz, host_clist.grid_size.z));
                    int neighbor_cell_id = pack_cell_3(neighbor_cell, host_clist.grid_size);
                    int neighbor_cell_start = host_clist.cell_start[neighbor_cell_id];
                    int neighbor_cell_end = host_clist.cell_end[neighbor_cell_id];
                    printf("    Neighbor cell (%d, %d, %d) [ID %d]: start=%d, end=%d\n", 
                        neighbor_cell.x, neighbor_cell.y, neighbor_cell.z,
                        neighbor_cell_id, neighbor_cell_start, neighbor_cell_end);
                }
            }
        }
    }
    return 0;
}