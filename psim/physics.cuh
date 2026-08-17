#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "helper_math.cuh"
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>
using namespace std;

struct particle
{
    float4 position;
    float4 force;
    float4 polarity;
};

__host__ __device__ inline float4 lj_12_6_c_forces(float4 r_vec,  float sigma, float epsilon, float r_c)
{
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float r2 = dot(r_vec, r_vec);
    float r_c2 = r_c * r_c;
    if (r2 > r_c2)
    {
        return force;
    }
    float r6 = r2 * r2 * r2;
    float r12 = r6 * r6;
    
};