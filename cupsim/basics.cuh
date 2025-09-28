#pragma once
#include <cuda_runtime.h>
#include "helper_math.h"

struct System
{
    float3 domain_min;
    float3 domain_max;
    int3 grid_size;
    float cell_size;
    float cutoff;
    bool use_periodic;
};

struct Particle
{
    float3 position;
    float3 polarity;
    float3 velocity;
    int id;
};

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


