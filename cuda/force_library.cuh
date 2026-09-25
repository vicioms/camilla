#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__ __forceinline__ float3 morse_force_3d(float3 r_vec, 
                        float r, 
                        float r_ref, 
                        float alpha, 
                        float V0, 
                        float r_cutoff)
{
    if (r > r_cutoff)
        return make_float3(0.0f, 0.0f, 0.0f);
    float exp_term = expf(-alpha * (r - r_ref));
    float exp_term_cutoff = expf(-alpha * (r_cutoff - r_ref));
    float force_magnitude = 2.0f * alpha * V0 * (exp_term * exp_term - exp_term);
    float force_cutoff = 2.0f * alpha * V0 * (exp_term_cutoff * exp_term_cutoff - exp_term_cutoff);
    float total_force_magnitude = force_magnitude - force_cutoff;
    return make_float3(total_force_magnitude * r_vec.x / r,
                       total_force_magnitude * r_vec.y / r,
                       total_force_magnitude * r_vec.z / r);
};

__host__ __device__ __forceinline__ float3 wca_force_3d(float3 r_vec, float r, float r_cutoff, float epsilon)
{
    if (r >= r_cutoff)
        return make_float3(0.0f, 0.0f, 0.0f);

    const float x   = r_cutoff / r;
    const float x2  = x * x;
    const float x6  = x2 * x2 * x2;

    // Since sigma = r_cutoff / 2^(1/6), we have (sigma/r)^6 = 0.5 * (r_cutoff/r)^6
    const float sr6 = 0.5f * x6;

    const float force_factor = 24.0f * epsilon * (2.0f * sr6 * sr6 - sr6) / (r * r);

    return make_float3(force_factor * r_vec.x, force_factor * r_vec.y, force_factor * r_vec.z);
};

__host__ __device__ __forceinline__
float3 lj_shifted_force_3d(float3 r_vec,
                           float r,
                           float sigma,
                           float r_cutoff,
                           float epsilon)
{
    if (r >= r_cutoff)
        return make_float3(0.0f, 0.0f, 0.0f);

    const float sr  = sigma / r;
    const float sr2 = sr * sr;
    const float sr6 = sr2 * sr2 * sr2;

    const float src  = sigma / r_cutoff;
    const float src2 = src * src;
    const float src6 = src2 * src2 * src2;

    const float force_factor =
        24.0f * epsilon *
        (
            (2.0f * sr6 * sr6 - sr6) / (r * r)
            -
            (2.0f * src6 * src6 - src6) / (r_cutoff * r)
        );

    return make_float3(force_factor * r_vec.x,
                       force_factor * r_vec.y,
                       force_factor * r_vec.z);
};




