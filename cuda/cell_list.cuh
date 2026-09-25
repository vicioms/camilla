#pragma once
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "math_funcs.cuh"
#include "cub_functions.cuh"
#include "generic_funcs.cuh"
#include "generic_kernels.cuh"

#include <algorithm>
#include <cassert>
#include <stdexcept>
using namespace std;

struct cell_list_3d
{
    int num_particles = 0;
    int num_cells = 0;

    float3 cell_size;
    float3 box_size;
    int3 grid_size;

    // Particle data
    float4* positions = nullptr;
    float4* sorted_positions = nullptr;
    float4* forces = nullptr;



    // Sorting permutation
    int* dummy_indices = nullptr;          // 0, ..., N - 1
    int* sorted_dummy_indices = nullptr;

    // Cell id of each particle
    int* pcell_ids = nullptr;
    int* sorted_pcell_ids = nullptr ;

    // Occupied-cell data
    int* cell_ids = nullptr;
    int* cell_offsets = nullptr;
    int* cell_counts = nullptr;

    int* num_unique_cells = nullptr;
    int h_num_unique_cells = 0;

    // Dense lookup over all cells
    int* cell_start = nullptr;
    int* cell_end = nullptr;

    // CUB temporary storage
    cub_workspace sorting_ws;
    cub_workspace unique_ws;
    cub_workspace esum_ws;
};


void device_reset_cell_list_3d(cell_list_3d& cl, int num_threads)
{
    if (num_threads <= 0)
        throw std::invalid_argument("num_threads must be positive");
    int num_particles = cl.num_particles;
    int num_cells = cl.num_cells;

    int num_blocks = (num_particles + num_threads - 1) / num_threads;
    _arithmetic_progression<<<num_blocks, num_threads>>>(
        cl.dummy_indices,
        0,
        1,
        num_particles
    );
    
    // redundant
    _arithmetic_progression<<<num_blocks, num_threads>>>(
        cl.sorted_dummy_indices,
        0,
        1,
        num_particles
    );
    
    _fill<<<num_blocks, num_threads>>>(
        cl.pcell_ids,
        -1,
        num_particles
    );
    _fill<<<num_blocks, num_threads>>>(
        cl.sorted_pcell_ids,
        -1,
        num_particles
    );
    _fill<<<num_blocks, num_threads>>>(
        cl.cell_ids,
        -1,
        num_particles
    );
    _fill<<<num_blocks, num_threads>>>(
        cl.cell_offsets,
        -1,
        num_particles
    );
    _fill<<<num_blocks, num_threads>>>(
        cl.cell_counts,
        0,
        num_particles
    );

    cl.h_num_unique_cells = 0;
    cudaMemset(cl.num_unique_cells, 0, sizeof(int));

    int num_blocks_cells = (cl.num_cells + num_threads - 1) / num_threads;
    _fill<<<num_blocks_cells, num_threads>>>(
        cl.cell_start,
        -1,
        num_cells
    );
    _fill<<<num_blocks_cells, num_threads>>>(
        cl.cell_end,
        -1,
        num_cells
    );

    
};

void device_allocate_cell_list_3d(cell_list_3d& cl, int num_particles, float3 cell_size, int3 grid_size, int num_threads)
{
    if (num_particles <= 0)
        throw std::invalid_argument("num_particles must be positive");
    if (cell_size.x <= 0.0f || cell_size.y <= 0.0f || cell_size.z <= 0.0f)
        throw std::invalid_argument("cell_size must be positive in all dimensions");
    if (grid_size.x <= 0 || grid_size.y <= 0 || grid_size.z <= 0)
        throw std::invalid_argument("grid_size must be positive in all dimensions");
    if (num_threads <= 0)
        throw std::invalid_argument("num_threads must be positive");
    
    cl.num_particles = num_particles;
    cl.cell_size = cell_size;
    cl.grid_size = grid_size;

    cl.box_size.x = cell_size.x * grid_size.x;
    cl.box_size.y = cell_size.y * grid_size.y;
    cl.box_size.z = cell_size.z * grid_size.z;

    cl.num_cells = cl.grid_size.x * cl.grid_size.y * cl.grid_size.z;

    cudaMalloc(&cl.positions, num_particles * sizeof(float4));
    cudaMalloc(&cl.sorted_positions, num_particles * sizeof(float4));

    cudaMalloc(&cl.forces, num_particles * sizeof(float4));

    cudaMalloc(&cl.dummy_indices, num_particles * sizeof(int));
    cudaMalloc(&cl.sorted_dummy_indices, num_particles * sizeof(int));

    cudaMalloc(&cl.pcell_ids, num_particles * sizeof(int));
    cudaMalloc(&cl.sorted_pcell_ids, num_particles * sizeof(int));

    // Allocate space for occupied-cell data
    // Note: we allocate space for all particles, but only the first num_unique_cells entries will be used
    cudaMalloc(&cl.cell_ids, cl.num_particles * sizeof(int));
    cudaMalloc(&cl.cell_offsets, cl.num_particles * sizeof(int));
    cudaMalloc(&cl.cell_counts, cl.num_particles * sizeof(int));

    cudaMalloc(&cl.num_unique_cells, sizeof(int));
    cl.h_num_unique_cells = 0;
    cudaMalloc(&cl.cell_start, cl.num_cells * sizeof(int));
    cudaMalloc(&cl.cell_end, cl.num_cells * sizeof(int));

    device_reset_cell_list_3d(cl, num_threads);
};

void device_init_random_lattice_cell_list_3d(cell_list_3d& cl, float jitter_amount, int seed)
{
    if (!(jitter_amount >= 0.0f && jitter_amount < 0.5f))
        throw std::invalid_argument("jitter_amount must be in [0, 0.5)");
    
    int num_particles = cl.num_particles;
    if (num_particles <= 0)
        throw std::invalid_argument("num_particles must be positive");
    
    float3 box_size = cl.box_size;

    default_random_engine rng(seed);
    int n_lattice = int(ceil(pow(num_particles, 1.0f / 3.0f)));
    int num_grid_points = n_lattice * n_lattice * n_lattice;
    assert(num_particles <= num_grid_points);
    float3 spacing = make_float3(
        box_size.x / n_lattice,
        box_size.y / n_lattice,
        box_size.z / n_lattice
    );
    int* grid_indices = new int[num_grid_points];
    for (int i = 0; i < num_grid_points; i++)
       grid_indices[i] = i;
    std::shuffle(grid_indices, grid_indices + num_grid_points, rng);
    uniform_real_distribution<float> jitter(-jitter_amount, jitter_amount);  // fraction of cell size
    float4* host_positions = new float4[num_particles]; 
    for (int i = 0; i < num_particles; i++)
    {
        int grid_index = grid_indices[i];
        int z = grid_index / (n_lattice * n_lattice);
        int y = (grid_index - z * n_lattice * n_lattice) / n_lattice;
        int x = grid_index % n_lattice;

        host_positions[i] = make_float4(
        (x + 0.5f + jitter(rng)) * spacing.x,
        (y + 0.5f + jitter(rng)) * spacing.y,
        (z + 0.5f + jitter(rng)) * spacing.z,
        0.0f);
    };
        
    delete[] grid_indices;
    cudaMemcpy(cl.positions, host_positions, num_particles * sizeof(float4), cudaMemcpyHostToDevice);
    delete[] host_positions;
};

void device_free_cell_list_3d(cell_list_3d& cl)
{
    cudaFree(cl.positions);
    cudaFree(cl.sorted_positions);
    cudaFree(cl.forces);
    cudaFree(cl.dummy_indices);
    cudaFree(cl.sorted_dummy_indices);
    cudaFree(cl.pcell_ids);
    cudaFree(cl.sorted_pcell_ids);
    cudaFree(cl.cell_ids);
    cudaFree(cl.cell_offsets);
    cudaFree(cl.cell_counts);
    cudaFree(cl.num_unique_cells);
    cudaFree(cl.cell_start);
    cudaFree(cl.cell_end);

    cl.sorting_ws.free();
    cl.unique_ws.free();
    cl.esum_ws.free();
};

void host_free_cell_list_3d(cell_list_3d& cl)
{
    delete[] cl.positions;
    delete[] cl.sorted_positions;
    delete[] cl.forces;
    delete[] cl.dummy_indices;
    delete[] cl.sorted_dummy_indices;
    delete[] cl.pcell_ids;
    delete[] cl.sorted_pcell_ids;
    delete[] cl.cell_ids;
    delete[] cl.cell_offsets;
    delete[] cl.cell_counts;
    delete[] cl.num_unique_cells;
    delete[] cl.cell_start;
    delete[] cl.cell_end;

    cl.sorting_ws.free();
    cl.unique_ws.free();
    cl.esum_ws.free();
};

inline void update_cell_list_3d(cell_list_3d& cl,int num_threads)
{
    if (num_threads <= 0)
        throw std::invalid_argument("num_threads must be positive");
    int num_particles = cl.num_particles;
    int num_cells = cl.num_cells;

    int3 grid_size = cl.grid_size;
    float3 cell_size = cl.cell_size;

    int num_blocks = (num_particles + num_threads - 1) / num_threads;


    // ------------------------------------------------------------
    // Compute cell id of each particle
    // ------------------------------------------------------------

    _pcell_ids_4_3d<<<num_blocks, num_threads>>>(
        cl.positions,
        cell_size,
        grid_size,
        cl.pcell_ids,
        num_particles
    );

    // Temporary diagnostics: distinguish failures preceding the CUB sort.
    cudaError_t cell_id_status = cudaGetLastError();
    if (cell_id_status != cudaSuccess)
        throw std::runtime_error(std::string("Cell-ID kernel launch: ") + cudaGetErrorString(cell_id_status));
    cell_id_status = cudaDeviceSynchronize();
    if (cell_id_status != cudaSuccess)
        throw std::runtime_error(std::string("Cell-ID kernel execution: ") + cudaGetErrorString(cell_id_status));

    // ------------------------------------------------------------
    // Sort particles by cell id
    // ------------------------------------------------------------

    cub_sort_pairs<int, int>(
        cl.sorting_ws,
        cl.pcell_ids,
        cl.sorted_pcell_ids,
        cl.dummy_indices,
        cl.sorted_dummy_indices,
        num_particles
    );


    // ------------------------------------------------------------
    // Reorder particle positions according to sorting permutation
    // ------------------------------------------------------------

    _reorder<<<num_blocks, num_threads>>>(
        cl.positions,
        cl.sorted_positions,
        cl.sorted_dummy_indices,
        num_particles
    );

    float4* tmp = cl.positions;
    cl.positions = cl.sorted_positions;
    cl.sorted_positions = tmp;


    // ------------------------------------------------------------
    // Find occupied cells and particle counts per cell
    // ------------------------------------------------------------

    cub_unique_counts(
        cl.unique_ws,
        cl.sorted_pcell_ids,
        cl.cell_ids,
        cl.cell_counts,
        cl.num_unique_cells,
        num_particles
    );


    // We need this number on the CPU to launch the following kernels
    cudaMemcpy(
        &cl.h_num_unique_cells,
        cl.num_unique_cells,
        sizeof(int),
        cudaMemcpyDeviceToHost
    );


    // ------------------------------------------------------------
    // Compute starting offset of each occupied cell
    // ------------------------------------------------------------

    cub_exclusive_sum(
        cl.esum_ws,
        cl.cell_counts,
        cl.cell_offsets,
        cl.h_num_unique_cells
    );


    // ------------------------------------------------------------
    // Dense cell -> [start, end) lookup
    // ------------------------------------------------------------

    cudaMemset(
        cl.cell_start,
        0xFF,
        num_cells * sizeof(int)
    );

    cudaMemset(
        cl.cell_end,
        0xFF,
        num_cells * sizeof(int)
    );


    int num_blocks_scatter =
        (cl.h_num_unique_cells + num_threads - 1) / num_threads;

    _scatter_start_end<<<num_blocks_scatter, num_threads>>>(
        cl.cell_ids,
        cl.cell_offsets,
        cl.cell_counts,
        cl.cell_start,
        cl.cell_end,
        cl.h_num_unique_cells
    );
};

void cell_list3d_to_host(cell_list_3d& dev_cl, cell_list_3d& host_cl)
{
    if (!dev_cl.positions || !dev_cl.sorted_positions || !dev_cl.forces || !dev_cl.dummy_indices || !dev_cl.sorted_dummy_indices || !dev_cl.pcell_ids || !dev_cl.sorted_pcell_ids || !dev_cl.cell_ids || !dev_cl.cell_counts || !dev_cl.cell_offsets || !dev_cl.cell_start || !dev_cl.cell_end) {
        throw std::invalid_argument("Invalid device cell list");
    }

    host_cl.num_particles = dev_cl.num_particles;
    host_cl.num_cells = dev_cl.num_cells;
    host_cl.cell_size = dev_cl.cell_size;
    host_cl.box_size = dev_cl.box_size;
    host_cl.grid_size = dev_cl.grid_size;
    host_cl.num_unique_cells = new int[1]{dev_cl.h_num_unique_cells};
    host_cl.h_num_unique_cells = dev_cl.h_num_unique_cells;

    host_cl.positions = new float4[dev_cl.num_particles];
    cudaMemcpy(host_cl.positions, dev_cl.positions, dev_cl.num_particles * sizeof(float4), cudaMemcpyDeviceToHost);
    host_cl.sorted_positions = new float4[dev_cl.num_particles];
    cudaMemcpy(host_cl.sorted_positions, dev_cl.sorted_positions, dev_cl.num_particles * sizeof(float4), cudaMemcpyDeviceToHost);

    host_cl.forces = new float4[dev_cl.num_particles];
    cudaMemcpy(host_cl.forces, dev_cl.forces, dev_cl.num_particles * sizeof(float4), cudaMemcpyDeviceToHost);

    host_cl.dummy_indices = new int[dev_cl.num_particles];
    cudaMemcpy(host_cl.dummy_indices, dev_cl.dummy_indices, dev_cl.num_particles * sizeof(int), cudaMemcpyDeviceToHost);
    host_cl.sorted_dummy_indices = new int[dev_cl.num_particles];
    cudaMemcpy(host_cl.sorted_dummy_indices, dev_cl.sorted_dummy_indices, dev_cl.num_particles * sizeof(int), cudaMemcpyDeviceToHost);

    host_cl.pcell_ids = new int[dev_cl.num_particles];
    cudaMemcpy(host_cl.pcell_ids, dev_cl.pcell_ids, dev_cl.num_particles * sizeof(int), cudaMemcpyDeviceToHost);
    host_cl.sorted_pcell_ids = new int[dev_cl.num_particles];
    cudaMemcpy(host_cl.sorted_pcell_ids, dev_cl.sorted_pcell_ids, dev_cl.num_particles * sizeof(int), cudaMemcpyDeviceToHost);

    host_cl.cell_ids = new int[dev_cl.h_num_unique_cells];
    cudaMemcpy(host_cl.cell_ids, dev_cl.cell_ids, dev_cl.h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_cl.cell_counts = new int[dev_cl.h_num_unique_cells];
    cudaMemcpy(host_cl.cell_counts, dev_cl.cell_counts, dev_cl.h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_cl.cell_offsets = new int[dev_cl.h_num_unique_cells];
    cudaMemcpy(host_cl.cell_offsets, dev_cl.cell_offsets, dev_cl.h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_cl.cell_start = new int[dev_cl.num_cells];
    cudaMemcpy(host_cl.cell_start, dev_cl.cell_start, dev_cl.num_cells * sizeof(int), cudaMemcpyDeviceToHost);

    host_cl.cell_end = new int[dev_cl.num_cells];
    cudaMemcpy(host_cl.cell_end, dev_cl.cell_end, dev_cl.num_cells * sizeof(int), cudaMemcpyDeviceToHost);
}
