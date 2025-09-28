#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <cmath>
using namespace std;

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

inline __device__ __host__ int pmod(int i, int n)
{
    return (i % n + n) % n;
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
}



inline __device__ __host__ int3 make_int3_periodic(int i, int j, int k, int3 grid_size) {
    return make_int3(pmod(i, grid_size.x), pmod(j, grid_size.y), pmod(k, grid_size.z));
};

inline __device__ __host__ int3 compute_cell(float3 pos,System system) {
    int3 cell;
    cell.x = (int)floorf((pos.x - system.domain_min.x) / system.cell_size);
    cell.y = (int)floorf((pos.y - system.domain_min.y) / system.cell_size);
    cell.z = (int)floorf((pos.z - system.domain_min.z) / system.cell_size);
    return cell;
};



inline __device__ __host__ int compute_cell_id(int3 cell, System system) {
    return cell.x + cell.y * system.grid_size.x + cell.z * system.grid_size.x * system.grid_size.y;
};

__global__ void compute_cells(Particle* particles, int3* cells, int* cell_ids, int num_particles, System system) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        float3 pos = particles[idx].position;
        int3 cell = compute_cell(pos, system);
        cells[idx] = cell;
        cell_ids[idx] = compute_cell_id(cell, system);
    }
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

void create_system(float3 domain_min, float3 domain_max, float cell_size, float cutoff, bool use_periodic, System& system)
{
    system.domain_min = domain_min;
    system.domain_max = domain_max;
    system.cell_size = cell_size;
    system.cutoff = cutoff;
    system.use_periodic = use_periodic;
    system.grid_size.x = (int)ceilf((domain_max.x - domain_min.x) / cell_size);
    system.grid_size.y = (int)ceilf((domain_max.y - domain_min.y) / cell_size);
    system.grid_size.z = (int)ceilf((domain_max.z - domain_min.z) / cell_size);
};














