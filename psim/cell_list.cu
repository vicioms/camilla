#include "cell_list.cuh"

__global__ void scatter_start_end(
    const int* __restrict__ indices,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    int* start,
    int* end,
    int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    int idx = indices[r];
    int offset = offsets[r];
    int count = counts[r];
    start[idx] = offset;
    end[idx] = offset + count;
}

__global__ void compute_pcell_ids_pbc(
    const particle* __restrict__ particles,
    const float3 cell_size,
    const int3 grid_size,
    int* cell_ids,
    int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float4 pos = particles[r].position;
    int3 cell_id;
    cell_id.x = pmod(pos.x / cell_size.x, grid_size.x);
    cell_id.y = pmod(pos.y / cell_size.y, grid_size.y);
    cell_id.z = pmod(pos.z / cell_size.z, grid_size.z);
    cell_ids[r] = pack_cell_3(cell_id, grid_size);
}

void cell_list::update_list_pbc(int num_threads_per_block)
{
    int num_blocks = (num_particles + num_threads_per_block - 1) / num_threads_per_block;
    compute_pcell_ids_pbc<<<num_blocks, num_threads_per_block>>>(
        particles, cell_size, grid_size, pcell_ids, num_particles);

    cub_sort_pairs<int, particle>(sorting_ws,
        pcell_ids, sorted_pcell_ids,
        particles, temp_particles,
        num_particles);

    particle* tmp = particles;
    particles = temp_particles;
    temp_particles = tmp;

    cub_unique_counts(unique_ws,
        sorted_pcell_ids, cell_ids, cell_counts,
        num_unique_cells, num_particles);

    cudaMemcpy(&h_num_unique_cells, num_unique_cells, sizeof(int), cudaMemcpyDeviceToHost);

    cub_exclusive_sum(esum_ws, cell_counts, cell_offsets, h_num_unique_cells);

    cudaMemset(cell_start, 0xFF, num_cells * sizeof(int));
    cudaMemset(cell_end, 0xFF, num_cells * sizeof(int));

    int num_blocks_scatter = (h_num_unique_cells + num_threads_per_block - 1) / num_threads_per_block;
    scatter_start_end<<<num_blocks_scatter, num_threads_per_block>>>(
        cell_ids, cell_offsets, cell_counts,
        cell_start, cell_end, h_num_unique_cells);
}

void cell_list::to_host(cell_list& host_out)
{
    host_out.num_particles = num_particles;
    host_out.num_cells = num_cells;
    host_out.cell_size = cell_size;
    host_out.box_size = box_size;
    host_out.grid_size = grid_size;
    host_out.num_unique_cells = new int[1]{h_num_unique_cells};
    host_out.h_num_unique_cells = h_num_unique_cells;

    host_out.particles = new particle[num_particles];
    cudaMemcpy(host_out.particles, particles, num_particles * sizeof(particle), cudaMemcpyDeviceToHost);

    host_out.cell_ids = new int[h_num_unique_cells];
    cudaMemcpy(host_out.cell_ids, cell_ids, h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_out.cell_counts = new int[h_num_unique_cells];
    cudaMemcpy(host_out.cell_counts, cell_counts, h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_out.cell_offsets = new int[h_num_unique_cells];
    cudaMemcpy(host_out.cell_offsets, cell_offsets, h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_out.cell_start = new int[num_cells];
    cudaMemcpy(host_out.cell_start, cell_start, num_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_out.cell_end = new int[num_cells];
    cudaMemcpy(host_out.cell_end, cell_end, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
}