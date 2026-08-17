#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

struct cub_workspace
{
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub_workspace() = default;

    void ensure_size(size_t required_bytes)
    {
        if (temp_storage_bytes < required_bytes)
        {
            if (temp_storage)
            {
                cudaFree(temp_storage);
            }
            cudaMalloc(&temp_storage, required_bytes);
            temp_storage_bytes = required_bytes;
        }
    };

    void free()
    {
        if (temp_storage)
        {
            cudaFree(temp_storage);
            temp_storage = nullptr;
            temp_storage_bytes = 0;
        }
    };


    ~cub_workspace()
    {
        free();
    }
};

template<typename K, typename V>
inline void cub_sort_pairs(cub_workspace& workspace, 
            K* keys_in, 
            K* keys_out, 
            V* values_in,
            V* values_out, 
            int num_items)
{
    size_t sort_required_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, 
        sort_required_bytes, 
        (K*)nullptr, 
        (K*)nullptr, 
        (V*)nullptr, 
        (V*)nullptr, 
        num_items);
    workspace.ensure_size(sort_required_bytes);
    cub::DeviceRadixSort::SortPairs(workspace.temp_storage, 
        workspace.temp_storage_bytes, 
        keys_in, 
        keys_out, 
        values_in, 
        values_out, 
        num_items);
};

inline void cub_unique_counts(cub_workspace& workspace,
            int* x, 
            int* x_unique,
            int* x_unique_counts,
            int* num_unique_out,
            int num_items)
{
    size_t required_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, 
        required_bytes, 
        (int*)nullptr, 
        (int*)nullptr, 
        (int*)nullptr,
        (int*)nullptr,
        num_items);
    workspace.ensure_size(required_bytes);
    cub::DeviceRunLengthEncode::Encode(workspace.temp_storage, 
        workspace.temp_storage_bytes, 
        x, 
        x_unique, 
        x_unique_counts,
        num_unique_out,
        num_items);
}

inline void cub_exclusive_sum(cub_workspace& workspace,
            int* x, 
            int* x_exclusive_sum, 
            int num_items)
{
    size_t required_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, 
        required_bytes, 
        (int*)nullptr, 
        (int*)nullptr, 
        num_items);
    workspace.ensure_size(required_bytes);
    cub::DeviceScan::ExclusiveSum(workspace.temp_storage, 
        workspace.temp_storage_bytes, 
        x, 
        x_exclusive_sum, 
        num_items);
};




