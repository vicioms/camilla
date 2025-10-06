#pragma once
#include <cuda_runtime.h>
#include "helper_math.h"

struct CubWorkspace
{
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CubWorkspace() = default;

    void ensure_size(size_t required_bytes)
    {
        if(temp_storage == nullptr || temp_storage_bytes < required_bytes) {
            if(temp_storage != nullptr) {
                cudaFree(temp_storage);
            }
            cudaMalloc(&temp_storage, required_bytes);
            temp_storage_bytes = required_bytes;
        }
    };

    void free()
    {
        if(temp_storage != nullptr) {
            cudaFree(temp_storage);
            temp_storage = nullptr;
            temp_storage_bytes = 0;
        }
    };

    ~CubWorkspace()
    {
        free();
    };
};


inline __device__ __host__ int pmod(int i, int n)
{
    return (i % n + n) % n;
};


inline __device__ __host__ int3 make_int3_periodic(int i, int j, int k, int3 grid_size) {
    return make_int3(pmod(i, grid_size.x), pmod(j, grid_size.y), pmod(k, grid_size.z));
};

inline __device__ __host__ int3 make_int3_periodic(int3 cell, int3 grid_size) {
    return make_int3(pmod(cell.x, grid_size.x), pmod(cell.y, grid_size.y), pmod(cell.z, grid_size.z));
};

// Wrap a scalar into [min, max)
__host__ __device__ __forceinline__
float wrap_pbc(float x, float minv, float maxv)
{
    float L = maxv - minv;          // box length (> 0)
    float r = x - minv;             // position relative to min
    // floorf handles negative r correctly; result in [0, L)
    r -= L * floorf(r / L);
    float w = minv + r;
    // Optional: nudge away from the top edge to avoid landing exactly at max
    // if (w == maxv) w = minv;
    return w;
};

__host__ __device__ __forceinline__
float3 wrap_pbc(float3 p, float3 minv, float3 maxv)
{
    float3 L = make_float3(maxv.x - minv.x,
                           maxv.y - minv.y,
                           maxv.z - minv.z);
    float rx = p.x - minv.x;  rx -= L.x * floorf(rx / L.x);  p.x = minv.x + rx;
    float ry = p.y - minv.y;  ry -= L.y * floorf(ry / L.y);  p.y = minv.y + ry;
    float rz = p.z - minv.z;  rz -= L.z * floorf(rz / L.z);  p.z = minv.z + rz;
    return p;
};

__host__ __device__ __forceinline__
float3 periodic_diff(float3 a, float3 b, float3 domain_min, float3 domain_max)
{
    const float3 box = domain_max - domain_min;
    float3 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z -= box.z * rintf(d.z / box.z);
    return d;
};

__host__ __device__ __forceinline__
float3 periodic_diff(float3 a, float3 b, float3 box)
{
    float3 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z -= box.z * rintf(d.z / box.z);
    return d;
};


inline void device_allocate(float3*& x, int n)
{
    cudaMalloc(&x, n * sizeof(float3));
};
inline void device_allocate(float*& x, int n)
{
    cudaMalloc(&x, n * sizeof(float));
};
inline void device_allocate(int3*& x, int n)
{
    cudaMalloc(&x, n * sizeof(int3));
};
inline void device_allocate(int*& x, int n)
{
    cudaMalloc(&x, n * sizeof(int));
};

inline __device__ __host__ bool morse_force(float3 r, float strength, float length_scale,  float r_0, float r_cutoff, float attractive_factor, float3& force, bool add)
{
    float dist2 = dot(r, r);
    if(dist2 < r_cutoff * r_cutoff) {
        float dist = sqrtf(dist2);
        if(dist < 1e-16f) {
            force = make_float3(0.0f, 0.0f, 0.0f);
            return false;
        };
        float exp_term = expf(-length_scale * (dist - r_0));
        float exp_at_cutoff = expf(-length_scale * (r_cutoff - r_0));
        float magnitude = 2.0f * strength * length_scale * (exp_term * exp_term - attractive_factor*exp_term);
        magnitude -= 2.0f * strength * length_scale * (exp_at_cutoff * exp_at_cutoff - attractive_factor*exp_at_cutoff);
        if(add) {
            force += magnitude * r/dist;
        }
        else
        {
            force = magnitude * r/dist;
        }
        return true;
    }
    else {
        force = make_float3(0.0f, 0.0f, 0.0f);
        return false;
    };
};