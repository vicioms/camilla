#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include "../helper_math.h"


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
                                float* __restrict__ force_integrator_z_q_imaginary,
                                float* __restrict__ q,
                                float* __restrict__ q2,
                                float* __restrict__ kernel_term_q,
                                float lambda_tilde,
                                float temp_tilde,
                                float mu_tilde,
                                float dt,
                                int Nk)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    if (kx >= Nk) return;
    float q_val = q[kx];
    float q2_val = q2[kx];
    float exp_arg = (dt * q2_val) * (1 + q2_val);
    field_integrator_q[kx] = expf(-exp_arg);
    noise_integrator_q[kx] = sqrtf(temp_tilde*(1 - expf(-2.0f*exp_arg))/(1+q2_val));
    force_integrator_q[kx] = lambda_tilde*kernel_term_q[kx]*(1 - field_integrator_q[kx]) / (1 + q2_val);
    force_integrator_z_q_imaginary[kx] = lambda_tilde * mu_tilde * q_val * kernel_term_q[kx];
};

__global__ void scale(float* __restrict__ x, float scale,  int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
};

__global__ void fill_noise(curandStatePhilox4_32_10_t* random_states,
                       float* __restrict__ noise,
                       int Ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Ntot) return;
    curandStatePhilox4_32_10_t local_state = random_states[idx];
    noise[idx] = curand_normal(&local_state);
    random_states[idx] = local_state; // save state back
};

__global__ void update(cplx* __restrict__ phi_q,
                        float* __restrict__ field_integrator_q, 
                        float* __restrict__ force_integrator_q, 
                        float* __restrict__ noise_integrator_q, 
                        cplx* __restrict__ noise_q,
                        int Nk)
{
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    if (kx >= Nk) return;
    cplx phi_q_val = phi_q[kx];
    cplx phi_q_val_next = make_cuComplex(field_integrator_q[kx] * phi_q_val.x,
                                                field_integrator_q[kx] * phi_q_val.y);
    phi_q_val_next.x += noise_integrator_q[kx] * noise_q[kx].x;
    phi_q_val_next.y += noise_integrator_q[kx] * noise_q[kx].y;
    phi_q[kx] = phi_q_val_next;
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

__global__ void device_float_to_complex(const float* __restrict__ in, cplx* __restrict__ out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        out[i] = make_cuComplex(in[i], 0.0f);
    }
};

void initialize_phi(float* out, float dx, int N)
{
    for(int i = 0; i < N; i++){
        out[i] = 1.0f/(dx*N);
    }
}


int main()
{
    int N = 8192*4;
    int Nk = N/2 + 1;
    int num_steps = 200000;
    cufftHandle plan_r2c, plan_c2r;
    cufftPlan1d(&plan_r2c, N, CUFFT_R2C, 1);
    cufftPlan1d(&plan_c2r, N, CUFFT_C2R, 1);

    int num_threads = 256;
    int num_blocks_N = (N+num_threads-1)/num_threads;
    int num_blocks_Nk = (Nk+num_threads-1)/num_threads;
    float dx = 0.01f;
    float dt = 0.001f;
    //params
    float lambda_tilde = 0.0f;
    float temp_tilde = 1.0f;
    float probe_radius = 0.1f;
    float mu_tilde = 1.0f;
    float timescales_factor = 100.0f;
    float velocities_ratio = 0.0f;

    //allocate qs
    float* q;
    float* q2;
    cudaMalloc(&q, sizeof(float)*Nk);
    cudaMalloc(&q2, sizeof(float)*Nk);
    make_qs<<<num_blocks_Nk, num_threads>>>(q, q2, dx, N);

    //allocate fields
    cplx* phi_q;
    float* z;
    //integrators
    float* field_integrator_q;
    float* force_integrator_q;
    float* noise_integrator_q;
    float* kernel_term_q;
    float* force_integrator_z_q_imaginary;
    float* force_z;
    //noise part
    curandStatePhilox4_32_10_t* random_states;
    float* noise;
    cplx* noise_q;
    cudaMalloc(&phi_q, sizeof(cplx) * Nk);
    cudaMalloc(&field_integrator_q, sizeof(float) * Nk);
    cudaMalloc(&force_integrator_q, sizeof(float) * Nk);
    cudaMalloc(&noise_integrator_q, sizeof(float) * Nk);
    cudaMalloc(&kernel_term_q, sizeof(float) * Nk);
    cudaMalloc(&force_integrator_z_q_imaginary, sizeof(float) * Nk);
    cudaMalloc(&random_states, N * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&noise, sizeof(float) * N);
    cudaMalloc(&noise_q, sizeof(cplx) * Nk);
    cudaMalloc(&z, sizeof(float));
    cudaMalloc(&force_z, sizeof(float));
    
    //populate the gaussian kernel
    make_kernel_term_q<<<num_blocks_Nk, num_threads>>>(kernel_term_q, q, probe_radius, Nk);
    scale<<<num_blocks_Nk, num_threads>>>(kernel_term_q, 1.0f/dx, Nk);
    //make integrators
    make_integrator<<<num_blocks_Nk, num_threads>>>(field_integrator_q, force_integrator_q, noise_integrator_q, 
                                                    force_integrator_z_q_imaginary, q, q2, kernel_term_q,
                                                    lambda_tilde, temp_tilde, mu_tilde, dt, Nk);
    //initialize random states
    init_random_states<<<num_blocks_N, num_threads>>>(random_states, 1101252, N);
    float* phi_x_host = new float[N];
    initialize_phi(phi_x_host, dx, N);
    float* phi_x_device;
    cudaMalloc(&phi_x_device, sizeof(float) * N);
    cudaMemcpy(phi_x_device, phi_x_host, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    //fourier transform initial condition
    cufftExecR2C(plan_r2c, phi_x_device, phi_q);
    cudaDeviceSynchronize();
    std::ofstream outfile("probe_fluid_1d_output.bin", std::ios::out | std::ios::binary);
    
    for(int step = 0; step < num_steps; step++){
        //generate noise
        fill_noise<<<num_blocks_N, num_threads>>>(random_states, noise, N);
        
        //scale<<<num_blocks_N, num_threads>>>(noise, sqrtf(1.0f/dx), N);
        //fourier transform noise
        cufftExecR2C(plan_r2c, noise, noise_q);
        //update phi_q
        update<<<num_blocks_Nk, num_threads>>>(phi_q,
                                                field_integrator_q,
                                                force_integrator_q,
                                                noise_integrator_q,
                                                noise_q,
                                                Nk);
        cudaDeviceSynchronize();

        if(step > num_steps/2 && step % 100 == 0){
            cufftExecC2R(plan_c2r, phi_q, phi_x_device);
            scale<<<num_blocks_N, num_threads>>>(phi_x_device, 1.0f/N, N);
            cudaMemcpy(phi_x_host, phi_x_device, sizeof(float)*N, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            outfile.write(reinterpret_cast<char*>(phi_x_host), sizeof(float) * N);
        }
    };

    outfile.close();
    delete[] phi_x_host;

    //cplx* fourier_test_device;
    //cudaMalloc(&fourier_test_device, sizeof(cplx)*Nk);
    //device_float_to_complex<<<num_blocks_N, num_threads>>>(kernel_term_q, fourier_test_device, Nk);
    //float* test_device;
    //cudaMalloc(&test_device, sizeof(float)*N);
    //cufftExecC2R(plan_c2r, fourier_test_device, test_device);
    //float* test_host = new float[N];
    //cudaMemcpy(test_host, test_device, sizeof(float)*N, cudaMemcpyDeviceToHost);
    //for(int i = 0; i < 20; i++){
    //    printf("%f\n", test_host[i]/N);
    //}
    //delete[] test_host;
    //cudaFree(test_device);
    //cudaFree(fourier_test_device);
    return 0;
};