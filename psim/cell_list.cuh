#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "helper_math.cuh"
#include "basic_funcs.cuh"
#include "basic_utils.cuh"
#include "physics.cuh"
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>
using namespace std;





struct cell_list
{
    int num_particles;
    int num_cells;
    float3 cell_size;
    float3 box_size;
    int3 grid_size;
    particle* particles;
    particle* temp_particles;
    int* pcell_ids;
    int* sorted_pcell_ids;
    int* cell_ids;
    int* cell_counts;
    int* cell_offsets;
    int* num_unique_cells;
    int h_num_unique_cells;
    int* cell_start;
    int* cell_end;
    cub_workspace sorting_ws;
    cub_workspace unique_ws;
    cub_workspace esum_ws;

    cell_list(int n, float3 csize, float3 bsize)
    {
        num_particles = n;
        cell_size = csize;
        box_size = bsize;
        grid_size.x = int(ceil(box_size.x / cell_size.x));
        grid_size.y = int(ceil(box_size.y / cell_size.y));
        grid_size.z = int(ceil(box_size.z / cell_size.z));
        num_cells = grid_size.x * grid_size.y * grid_size.z;
        cudaMalloc(&particles, num_particles * sizeof(particle));
        cudaMalloc(&temp_particles, num_particles * sizeof(particle));
        cudaMalloc(&pcell_ids, num_particles * sizeof(int));
        cudaMalloc(&sorted_pcell_ids, num_particles * sizeof(int));
        cudaMalloc(&cell_ids, num_particles * sizeof(int));
        cudaMalloc(&cell_counts, num_particles * sizeof(int));
        cudaMalloc(&cell_offsets, num_particles * sizeof(int));
        cudaMalloc(&num_unique_cells, sizeof(int));
        cudaMalloc(&cell_start, num_cells * sizeof(int));
        cudaMalloc(&cell_end, num_cells * sizeof(int));
    }

    void update_list_pbc(int num_threads_per_block);
    void to_host(cell_list& host_clist);
};

