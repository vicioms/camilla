#include "base.cuh"

__global__ void coo_scale(float* values,  int num_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_values)
        return;
    values[n] *= scale;
};

__global__ void coo_mul(float* mat_values, int* __restrict__ row_indices, int* __restrict__  col_indices, 
                        float* vector_in, float* vector_out, int num_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_values)
        return;
    int row = row_indices[idx];
    int col = col_indices[idx];
    atomicAdd(&vector_out[row], mat_values[idx]*vector_in[col]);
    return;
};