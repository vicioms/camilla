# pragma once
#include <cuda.h>
#include <cuda_runtime.h>


// ======================
// vector kernels 
// ======================

template<typename T>
__global__ void _fill(T* x, T value, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    x[i] = value;
};

template<typename T>
__global__ void _arithmetic_progression(
    T* x,
    T start,
    T step,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    x[i] = start + static_cast<T>(i) * step;
};



// ======================
// utility kernels 
// ======================

__global__ void _scatter_start_end(
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
};

template<typename T>
__global__ void _reorder(
    const T* x_in,
    T* x_out,
    const int* indices_sorted,
    int num_items)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_items)
        return;

    x_out[i] = x_in[indices_sorted[i]];
}

template<typename T1, typename T2>
__global__ void _reorder2(
    const T1* x_in,
    const T2* y_in,
    T1* x_out,
    T2* y_out,
    const int* indices_sorted,
    int num_items)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_items)
        return;

    x_out[i] = x_in[indices_sorted[i]];
    y_out[i] = y_in[indices_sorted[i]];
}

template<typename T1, typename T2, typename T3>
__global__ void _reorder3(
    const T1* x_in,
    const T2* y_in,
    const T3* z_in,
    T1* x_out,
    T2* y_out,
    T3* z_out,
    const int* indices_sorted,
    int num_items)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_items)
        return;

    x_out[i] = x_in[indices_sorted[i]];
    y_out[i] = y_in[indices_sorted[i]];
    z_out[i] = z_in[indices_sorted[i]];
};




// ======================
// cell list kernels
// ======================
__global__ void _pcell_ids_4_2d(
    const float4* __restrict__ positions,
    const float2 cell_size,
    const int2 grid_size,
    int* pcell_ids,
    int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= n)
        return;

    float4 pos = positions[r];

    int2 cell_id;

    cell_id.x = pmod((int)floorf(pos.x / cell_size.x), grid_size.x);
    cell_id.y = pmod((int)floorf(pos.y / cell_size.y), grid_size.y);
    
    pcell_ids[r] = pack_cell_2d(cell_id, grid_size);
};

__global__ void _pcell_ids_4_3d(
    const float4* __restrict__ positions,
    const float3 cell_size,
    const int3 grid_size,
    int* pcell_ids,
    int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= n)
        return;

    float4 pos = positions[r];

    int3 cell_id;

    cell_id.x = pmod((int)floorf(pos.x / cell_size.x), grid_size.x);
    cell_id.y = pmod((int)floorf(pos.y / cell_size.y), grid_size.y);
    cell_id.z = pmod((int)floorf(pos.z / cell_size.z), grid_size.z);
    
    pcell_ids[r] = pack_cell_3d(cell_id, grid_size);
};







