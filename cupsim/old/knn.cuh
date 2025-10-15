#pragma once
#include "helper_math.h"

template<int K>
struct top_k
{
    float values[K];
    int indices[K];
    int max_pos;
    int num_occupied;

    __device__ void init(float inf=CUDART_INF_F) {
        #pragma unroll
        for (int i=0;i<K;++i){ values[i]=inf; indices[i]=-1; }
        max_pos = 0;
        num_occupied = 0;
    };

    __device__ void try_insert(float v, float idx)
    {
        if(idx < 0) return;
        if(v >= values[max_pos]) return;
        values[max_pos] = v;
        indices[max_pos] = idx;
        int m = 0;
        #pragma unroll
        for (int k=1;k<K;++k) if (values[k] > values[m]) m=k; 
        max_pos = m;
        if(num_occupied < K) num_occupied++;
    };

    __device__ void load_from_global(const int* indices_to_load, 
                                    const float* values_to_load,
                                    int q)
    {
        #pragma unroll
        for(int k = 0; k < K; ++k) {
            indices[k] = indices_to_load[q*K + k];
            values[k] = values_to_load[q*K + k];
        };
        int m = 0;
        #pragma unroll
        for (int k=1;k<K;++k) if (values[k] > values[m]) m=k; max_pos=m;
        num_occupied = K;
    };

    __device__ void store_to_global(int* indices_to_store, 
                                    float* values_to_store,
                                    int q) const
    {
        #pragma unroll
        for(int k = 0; k < K; ++k) {
            indices_to_store[q*K + k] = indices[k];
            values_to_store[q*K + k] = values[k];
        };
    };
};

template<int K>
__global__ void knn_merge_tile(
    const float* __restrict__ G,
    const float* __restrict__ nx,
    const float* __restrict__ ny,
    int q_start, int n_start,
    int Qb, int Nb,
    int* __restrict__ out_indices,
    float* __restrict__ out_values)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Qb) return;

    const int q = q_start + r;
    top_k<K> tk;
    tk.init();
    tk.load_from_global(out_indices, out_values, q);
    //this fetches the r-th row of the tile
    const float* dot_products = G + r*Nb;
    const float nxv = nx[q];

    // scan the columns of this tile
    for (int c=0; c<Nb; ++c) {
        float d2 = dot_products[c] + nxv + ny[n_start + c];
        tk.try_insert(d2, n_start + c);
    };
    tk.store_to_global(out_indices, out_values, q);
};




