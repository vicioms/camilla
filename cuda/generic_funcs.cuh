#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <string>
#include <stdexcept>
#include "math_funcs.cuh"

using namespace std;

// ======================
// Generic
// ======================

inline void cuda_check(cudaError_t error)
{
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}

 __device__ __host__ __forceinline__ int pmod(int i, int n)
{
    return (i % n + n) % n;
};


// ======================
// Periodic boundary conditions
// ======================


__host__ __device__ __forceinline__ float4 wrap4_3d(float4 x, float4 L)
{
    return make_float4(fmodf(fmodf(x.x, L.x) + L.x, L.x),
                       fmodf(fmodf(x.y, L.y) + L.y, L.y),
                       fmodf(fmodf(x.z, L.z) + L.z, L.z),
                       0.0f);
};
__host__ __device__ __forceinline__ float4 wrap4_2d(float4 x, float4 L)
{
    return make_float4(fmodf(fmodf(x.x, L.x) + L.x, L.x),
                       fmodf(fmodf(x.y, L.y) + L.y, L.y),
                       0.0f,
                       0.0f);
};
__host__ __device__ __forceinline__ float3 wrap_3d(float3 x, float3 L)
{
    return make_float3(fmodf(fmodf(x.x, L.x) + L.x, L.x),
                       fmodf(fmodf(x.y, L.y) + L.y, L.y),
                       fmodf(fmodf(x.z, L.z) + L.z, L.z));
};
__host__ __device__ __forceinline__ float2 wrap_2d(float2 x, float2 L)
{
    return make_float2(fmodf(fmodf(x.x, L.x) + L.x, L.x),
                       fmodf(fmodf(x.y, L.y) + L.y, L.y));
};
__host__ __device__ __forceinline__ float4 wdiff4_3d(float4 a, float4 b, float4 box)
{
    float4 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z -= box.z * rintf(d.z / box.z);
    d.w = 0.0f;
    return d;
};
__host__ __device__ __forceinline__ float4 wdiff4_2d(float4 a, float4 b, float4 box)
{
    float4 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z = 0.0f;
    d.w = 0.0f;
    return d;
};
__host__ __device__ __forceinline__ float3 wdiff_3d(float3 a, float3 b, float3 box)
{
    float3 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z -= box.z * rintf(d.z / box.z);
    return d;
};
__host__ __device__ __forceinline__ float2 wdiff_2d(float2 a, float2 b, float2 box)
{
    float2 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    return d;
};

// ======================
// Cell packing/unpacking
// ======================

__host__ __device__ __forceinline__ int pack_cell_2d(int2 cell, int2 grid_size)
{
    return cell.x + cell.y * grid_size.x;
};
__host__ __device__ __forceinline__ int pack_cell_3d(int3 cell, int3 grid_size)
{
    return cell.x + cell.y * grid_size.x + cell.z * grid_size.x * grid_size.y;
};
__host__ __device__ __forceinline__ int3 unpack_cell_3d(int cell_id, int3 grid_size)
{
    int x = cell_id % grid_size.x;
    int y = (cell_id / grid_size.x) % grid_size.y;
    int z = cell_id / (grid_size.x * grid_size.y);
    return make_int3(x, y, z);
};
__host__ __device__ __forceinline__ int2 unpack_cell_2d(int cell_id, int2 grid_size)
{
    int x = cell_id % grid_size.x;
    int y = (cell_id / grid_size.x) % grid_size.y;
    return make_int2(x, y);
};
