#pragma once
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <string>

struct cub_workspace
{
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub_workspace() = default;

    cub_workspace(const cub_workspace&) = delete;
    cub_workspace& operator=(const cub_workspace&) = delete;

    cub_workspace(cub_workspace&& other) noexcept
        : temp_storage(other.temp_storage),
          temp_storage_bytes(other.temp_storage_bytes)
    {
        other.temp_storage = nullptr;
        other.temp_storage_bytes = 0;
    }

    cub_workspace& operator=(cub_workspace&& other) noexcept
    {
        if (this != &other)
        {
            free();

            temp_storage = other.temp_storage;
            temp_storage_bytes = other.temp_storage_bytes;

            other.temp_storage = nullptr;
            other.temp_storage_bytes = 0;
        }

        return *this;
    }

    void ensure_size(size_t required_bytes)
    {
        if (temp_storage_bytes >= required_bytes)
            return;

        if (temp_storage)
            cudaFree(temp_storage);

        cudaMalloc(&temp_storage, required_bytes);
        temp_storage_bytes = required_bytes;
    }

    void free()
    {
        if (temp_storage)
        {
            cudaFree(temp_storage);
            temp_storage = nullptr;
            temp_storage_bytes = 0;
        }
    }

    ~cub_workspace()
    {
        free();
    }
};

template<typename K, typename V>
inline void cub_sort_pairs(
    cub_workspace& workspace,
    K* keys_in,
    K* keys_out,
    V* values_in,
    V* values_out,
    int num_items)
{
    size_t required_bytes = 0;
    cudaError_t cuda_status = cub::DeviceRadixSort::SortPairs(
        nullptr,
        required_bytes,
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_items
    );
    if (cuda_status != cudaSuccess)
        throw std::runtime_error(
            std::string("CUB SortPairs storage query: ") + cudaGetErrorString(cuda_status));

    workspace.ensure_size(required_bytes);

    cuda_status = cub::DeviceRadixSort::SortPairs(
        workspace.temp_storage,
        workspace.temp_storage_bytes,
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_items
    );

    if (cuda_status != cudaSuccess)
        throw std::runtime_error(
            std::string("CUB SortPairs execution: ") + cudaGetErrorString(cuda_status));
}

inline void cub_unique_counts(
    cub_workspace& workspace,
    int* x,
    int* x_unique,
    int* x_unique_counts,
    int* num_unique_out,
    int num_items)
{
    size_t required_bytes = 0;

    cudaError_t cuda_status = cub::DeviceRunLengthEncode::Encode(
        nullptr,
        required_bytes,
        x,
        x_unique,
        x_unique_counts,
        num_unique_out,
        num_items
    );
    if (cuda_status != cudaSuccess)
        throw std::runtime_error(
            std::string("CUB Encode storage query: ") + cudaGetErrorString(cuda_status));

    workspace.ensure_size(required_bytes);

    cuda_status = cub::DeviceRunLengthEncode::Encode(
        workspace.temp_storage,
        workspace.temp_storage_bytes,
        x,
        x_unique,
        x_unique_counts,
        num_unique_out,
        num_items
    );

    if (cuda_status != cudaSuccess)
        throw std::runtime_error(
            std::string("CUB Encode execution: ") + cudaGetErrorString(cuda_status));
}

inline void cub_exclusive_sum(
    cub_workspace& workspace,
    int* x,
    int* x_exclusive_sum,
    int num_items)
{
    size_t required_bytes = 0;

    cudaError_t cuda_status = cub::DeviceScan::ExclusiveSum(
        nullptr,
        required_bytes,
        x,
        x_exclusive_sum,
        num_items
    );

    if (cuda_status != cudaSuccess)
        throw std::runtime_error(
            std::string("CUB ExclusiveSum storage query: ") + cudaGetErrorString(cuda_status));

    workspace.ensure_size(required_bytes);

    cuda_status = cub::DeviceScan::ExclusiveSum(
        workspace.temp_storage,
        workspace.temp_storage_bytes,
        x,
        x_exclusive_sum,
        num_items
    );
    if (cuda_status != cudaSuccess)
        throw std::runtime_error(
            std::string("CUB ExclusiveSum execution: ") + cudaGetErrorString(cuda_status));
}
