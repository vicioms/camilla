#pragma once
#include <cuda_runtime.h>
#include "helper_math.cuh"


inline __device__ __host__ int pmod(int i, int n)
{
    return (i % n + n) % n;
};

__host__ __device__ __forceinline__ float4 wrap_4_3(float4 x, float4 L)
{
    return make_float4(fmodf(fmodf(x.x, L.x) + L.x, L.x),
                       fmodf(fmodf(x.y, L.y) + L.y, L.y),
                       fmodf(fmodf(x.z, L.z) + L.z, L.z),
                       0.0f);
};

__host__ __device__ __forceinline__ float4 wdiff_4_2(float4 a, float4 b, float2 box)
{
    float4 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z = 0.0f;
    d.w = 0.0f;
    return d;
};

__host__ __device__ __forceinline__ float4 wdiff_4_3(float4 a, float4 b, float3 box)
{
    float4 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z -= box.z * rintf(d.z / box.z);
    d.w = 0.0f;
    return d;
};


__host__ __device__ __forceinline__ int3 unpack_cell_pbc_3(int cell_id, int3 grid_size)
{
    int z = pmod(cell_id, grid_size.z);
    int y = pmod((cell_id - z) / grid_size.z, grid_size.y);
    int x = pmod((cell_id - z - y * grid_size.z) / (grid_size.z * grid_size.y), grid_size.x);
    return make_int3(x, y, z);
};

__host__ __device__ __forceinline__ int pack_cell_3(int3 cell_idx, int3 grid_size)
{
    return cell_idx.x + cell_idx.y * grid_size.x + cell_idx.z * grid_size.x * grid_size.y;
};
