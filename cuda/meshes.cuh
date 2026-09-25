#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.cuh"
#include "helper_funcs.cuh"
#include "helper_cub.cuh"

struct mesh
{
    int num_vertices = 0;
    int num_edges = 0;
    int num_cells = 0;
    float3* vertices = nullptr;
    int2* edge_vertices = nullptr; // from to
    int2* edge_cells = nullptr; // left right cells
    int* cell_edges = nullptr; // edges of each cell
    int* cell_edge_offsets = nullptr; // offsets for cell_edges
};




