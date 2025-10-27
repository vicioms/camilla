#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include "helper_math.h"


using cplx = cufftComplex;

__host__ __device__ __forceinline__
float wrapped_float_diff(float a, float b, float L)
{
    float d = a - b;
    d -= L * rintf(d / L);
    return d;
};

__host__ __device__ __forceinline__
float wrap_float(float p, float origin, float size)
{
    float rx = p - origin;  
    rx -= size * floorf(rx / size);  
    p = origin + rx;
    return p;
};

__global__ void init_random_states(curandStatePhilox4_32_10_t* states,
                            unsigned long long seed, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // sequence = gid makes streams independent per thread
        curand_init(seed, /*sequence*/ gid, /*offset*/ 0, &states[gid]);
    }
};

__global__ void make_qs(float* __restrict__ q, float* __restrict__ q2, float dx, int N){
    int Nk = N / 2 + 1;
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    if (kx >= Nk) return;
    float q_val = 2.0f * M_PI * kx / (N * dx);
    q[kx] = q_val;
    q2[kx] = q_val * q_val;
};

__global__ void make_kernel_term_q(float* __restrict__ kernel_term_q, 
                            float* __restrict__ q,
                            float spatial_scale,
                            int Nk)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    if (kx >= Nk) return;
    float q_val = q[kx];
    kernel_term_q[kx] = expf(-0.5*(spatial_scale * q_val)*(spatial_scale * q_val));
};

__global__ void make_integrator(float* __restrict__ field_integrator_q, 
                                float* __restrict__ force_integrator_q, 
                                float* __restrict__ noise_integrator_q, 
                                float* __restrict__ q2,
                                float* __restrict__ kernel_term_q,
                                float temp_tilde,
                                float dt,
                                int Nk)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    if (kx >= Nk) return;
    float q2_val = q2[kx];
    float exp_arg = (dt * q2_val) * (1 + q2_val);
    field_integrator_q[kx] = expf(-exp_arg);
    noise_integrator_q[kx] = sqrtf(dt*(1 - expf(-2.0f*exp_arg)) * temp_tilde/(1+q2_val));
    force_integrator_q[kx] = kernel_term_q[kx]*(1 - field_integrator_q[kx]) / (1 + q2_val);
};

__global__ void scale(float* __restrict__ x, float scale,  int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
};

__global__ void update(cplx* __restrict__ phi_q,
                        float* __restrict__ field_integrator_q, 
                        float* __restrict__ force_integrator_q, 
                        float* __restrict__ noise_integrator_q, 
                        curandStatePhilox4_32_10_t* __restrict__ rand_states,
                        int Nk)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    if (kx >= Nk) return;
    cplx phi_q_val = phi_q[kx];
    cplx field_term = make_cuComplex(field_integrator_q[kx] * phi_q_val.x,
                                                field_integrator_q[kx] * phi_q_val.y);
    
};

float sum_on_cpu(float* x, int n){
    float sum = 0.0f;
    float* h_x = new float[n];
    cudaMemcpy(h_x, x, sizeof(float)*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++){
        sum += h_x[i];
    }
    delete[] h_x;
    return sum;
};

int main()
{
    int N = 8192;
    int Nk = N/2 + 1;
    int num_threads = 256;
    int num_blocks_N = (N+num_threads-1)/num_threads;
    int num_blocks_Nk = (Nk+num_threads-1)/num_threads;
    float dx = 0.01f;
    float dt = 0.001f;
    //params
    float temp_tilde = 0.1f;
    float probe_radius = 0.1f;
    //allocate qs
    float* q;
    float* q2;
    cudaMalloc(&q, sizeof(float)*Nk);
    cudaMalloc(&q2, sizeof(float)*Nk);
    make_qs<<num_blocks_Nk, num_threads>>>(q, q2, dx, N);
    //allocate fields
    cplx* phi_q;
    float* field_integrator_q;
    float* force_integrator_q;
    float* noise_integrator_q;
    float* kernel_term_q;
    cudaMalloc(&phi_q, sizeof(cplx) * Nk);
    cudaMalloc(&field_integrator_q, sizeof(float) * Nk);
    cudaMalloc(&force_integrator_q, sizeof(float) * Nk);
    cudaMalloc(&noise_integrator_q, sizeof(float) * Nk);
    cudaMalloc(&kernel_term_q, sizeof(float) * Nk);
    //populate the gaussian kernel
    make_kernel_term_q<<num_blocks_Nk, num_threads>>>(kernel_term_q, q, probe_radius, Nk);
    float normalization_kernel_term_q = sum_on_cpu(kernel_term_q, Nk);
    scale<<<num_blocks_Nk, num_threads>>>(kernel_term_q, 1.0f/normalization_kernel_term_q, Nk);
    //make integrators
    make_integrator<<num_blocks_Nk, num_threads>>>(field_integrator_q, force_integrator_q, noise_integrator_q,
                                                    q2, kernel_term_q,
                                                    temp_tilde, dt, Nk);

    return 0;
};