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
float2 wrapped_float2_diff(float2 a, float2 b, float2 box)
{
    float2 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    return d;
};

__host__ __device__ __forceinline__
float2 wrap_float2(float2 p, float2 box_origin, float2 box)
{
    float rx = p.x - box_origin.x;  rx -= box.x * floorf(rx / box.x);  p.x = box_origin.x + rx;
    float ry = p.y - box_origin.y;  ry -= box.y * floorf(ry / box.y);  p.y = box_origin.y + ry;
    return p;
};

__host__ __device__ __forceinline__ int idx2(int x, int y, int Nx) { return y * Nx + x; }


// ---- precompute k^2 on the R2C grid (Nxk = Nx/2+1 by Ny)
__global__ void kernel_make_k2(float* __restrict__ k2, int Nx, int Ny){
    int Nxk = Nx/2 + 1;
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    int ky = blockIdx.y * blockDim.y + threadIdx.y;   // 0..Ny-1
    if (kx >= Nxk || ky >= Ny) return;
    int id = idx2(kx, ky, Nxk);
    int kx_phys = kx;                                  // non-negative
    int ky_phys = (ky <= Ny/2) ? ky : ky - Ny;         // wrap negative freqs
    float k2v = float(kx_phys * kx_phys + ky_phys * ky_phys);
    //float k2v = 4 - 2 * cosf(2.0f * M_PI * kx_phys / Nx) - 2 * cosf(2.0f * M_PI * ky_phys / Ny); // finite difference Laplacian
    k2[id] = k2v;
};

// ---- semi-implicit update in Fourier space
__global__ void kernel_update_modes(const cplx* __restrict__ phi_k,
                                    const cplx* __restrict__ spatial_forces_k,
                                    const cplx* __restrict__ white_noise_k,
                                    const float* __restrict__ k2,
                                    cplx* __restrict__ phi_k_next,
                                    int Nx, int Ny, float mu, float kappa, float mobility, float temp, float dt){
    int Nxk = Nx/2 + 1;
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    if (kx >= Nxk || ky >= Ny) return;
    int id = idx2(kx, ky, Nxk);

    float k2v = k2[id];
    float denom = 1.0f + dt  * (mu * k2v + kappa * k2v * k2v)*mobility;

    cplx num = phi_k[id];
    num.x += (dt*k2v*mobility)*spatial_forces_k[id].x + (sqrtf(2*dt*mobility*temp*k2v)*white_noise_k[id].x);
    num.y += (dt*k2v*mobility)*spatial_forces_k[id].y + (sqrtf(2*dt*mobility*temp*k2v)*white_noise_k[id].y);
    phi_k_next[id].x = num.x / denom;
    phi_k_next[id].y = num.y / denom;
};

__global__ void init_random_states(curandStatePhilox4_32_10_t* states,
                            unsigned long long seed, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // sequence = gid makes streams independent per thread
        curand_init(seed, /*sequence*/ gid, /*offset*/ 0, &states[gid]);
    }
};

__global__ void kernel_fft_normalize(float* __restrict__ c, int Ntot, float invN){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ntot) c[i] *= invN;
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

__global__ void compute_spatial_forces(const float* __restrict__ phi,
                                   const float2* __restrict__ probe,
                                   float* __restrict__ forces,
                                   int Nx, int Ny, const float interaction_strength, const float probe_radius) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    int id = idx2(ix, iy, Nx);
    float2 probe_position = *probe;
    float2 r = wrapped_float2_diff(make_float2(ix, iy), probe_position, make_float2(Nx, Ny));
    float r_mod = sqrtf(r.x * r.x + r.y * r.y);
    float V = expf(- (r_mod * r_mod) / (2.0f * probe_radius * probe_radius));
    forces[id] = -interaction_strength * V;
};

__global__ void accumulate_probe_forces(const float* __restrict__ phi,
                                   const float2* __restrict__ probe,
                                   float2* __restrict__ probe_force,
                                   int Nx, int Ny, const float interaction_strength, const float probe_radius) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    int id = idx2(ix, iy, Nx);
    float2 probe_position = *probe;
    float2 r = wrapped_float2_diff(make_float2(ix, iy), probe_position, make_float2(Nx, Ny));
    float r_mod = sqrtf(r.x * r.x + r.y * r.y);
    float force_term = (interaction_strength/(probe_radius*probe_radius))*expf(- (r_mod * r_mod) / (2.0f * probe_radius * probe_radius));
    float field_value = phi[id];
    atomicAdd(&probe_force->x, field_value*force_term * r.x);
    atomicAdd(&probe_force->y, field_value*force_term * r.y);
};

__global__  void probe_step(__restrict__ float2* probe, __restrict__ float2* probe_force, float* u, float well_strength, float well_velocity, float well_y,  float dt, float t, int Nx, int Ny){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0) return;
    float2 current_probe = *probe;
    float2 current_force = *probe_force;
    float well_position_x =  well_velocity * t;
    well_position_x -= Nx * floorf(well_position_x / Nx); // wrap
    float well_gap_x = well_position_x - current_probe.x;
    well_gap_x -= Nx * rintf(well_gap_x / Nx); // minimum image
    *u = well_strength*well_gap_x; // output potential energy
    float well_gap_y = well_y - current_probe.y;
    //euler step
    current_probe.x += dt * current_force.x + well_strength * well_gap_x * dt;
    current_probe.y += dt * current_force.y + well_strength * well_gap_y * dt;
    *probe = wrap_float2(current_probe, make_float2(0.0f, 0.0f), make_float2(Nx, Ny));
    *probe_force = make_float2(0.0f, 0.0f); // reset force
};

void initialize_field(float* phi, float phi0_amplitude, int Nx, int Ny){
    std::mt19937 gen(42);
    std::normal_distribution<float> distr(0.0f, 1.0f);
    std::vector<float> phi_host(Nx * Ny);
    float phi_sum = 0.0f;
    for (int i = 0; i < Nx * Ny; ++i)
    {
        phi_host[i] = phi0_amplitude * distr(gen);
        phi_sum += phi_host[i];
    };
    float phi_mean = phi_sum / float(Nx * Ny);
    for (int i = 0; i < Nx * Ny; ++i)
    {
        phi_host[i] -= phi_mean;
    };
    cudaMemcpy(phi, phi_host.data(), phi_host.size() * sizeof(float), cudaMemcpyHostToDevice);
}




int main(){
    // --- grid
    const int Nx = 2048;                 // tune
    const int Ny = 64;
    const int Ntot = Nx * Ny;
    const int Nxk = Nx/2 + 1;
    const size_t Nk = size_t(Ny) * size_t(Nxk); // (Ny, Nx/2+1)

    // --- physics
    const float dt   = 1e-2f;
    const float mu = 1.0f;
    const float kappa = 25.0f;
    const float mobility = 0.01f;
    const float temp = 10.0f;
    const float interaction_strength = 1.0f;
    const float probe_radius = 1.0f;
    const float well_strength = 0.01f;
    const float well_velocity = 1.0f;
    const float well_y = Ny / 2.0f;
    const float invN = 1.0f / float(Nx * Ny);
    const float phi0_amplitude = 100.0f;
    float2 probe0 = make_float2(0.0, well_y);
    float2 force0 = make_float2(0.0f, 0.0f);

    float2* probe;
    float2* probe_force;
    float* u;
    cudaMalloc(&probe, sizeof(float2));
    cudaMalloc(&u, sizeof(float));
    cudaMalloc(&probe_force, sizeof(float2));
    cudaMemcpy(probe, &probe0, sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(probe_force, &force0, sizeof(float2), cudaMemcpyHostToDevice);

    float *phi;
    float *noise;
    float *spatial_forces;
    cudaMalloc(&phi,  Ntot * sizeof(float));
    cudaMalloc(&noise, Ntot * sizeof(float));
    cudaMalloc(&spatial_forces, Ntot * sizeof(float));

    cplx* phi_k, *phi_k_next;
    cplx* noise_k;
    cplx* spatial_forces_k;
    cudaMalloc(&phi_k,      Nk * sizeof(cplx));
    cudaMalloc(&phi_k_next, Nk * sizeof(cplx));
    cudaMalloc(&noise_k,    Nk * sizeof(cplx));
    cudaMalloc(&spatial_forces_k, Nk * sizeof(cplx));

    float *k2;
    cudaMalloc(&k2, Nk * sizeof(float));
    curandStatePhilox4_32_10_t* random_states;
    cudaMalloc(&random_states, Ntot * sizeof(curandStatePhilox4_32_10_t));
    

    initialize_field(phi, phi0_amplitude, Nx, Ny);

    cufftHandle plan_r2c, plan_c2r;
    cufftPlan2d(&plan_r2c, Ny, Nx, CUFFT_R2C);
    cufftPlan2d(&plan_c2r, Ny, Nx, CUFFT_C2R);

    dim3 block2d(16, 16);
    dim3 grid_float((Nx + block2d.x - 1)/block2d.x,
                   (Ny + block2d.y - 1)/block2d.y);
    dim3 grid_k((Nxk + block2d.x - 1)/block2d.x,
                (Ny  + block2d.y - 1)/block2d.y);
    int tpb = 256;
    int nb  = (Ntot + tpb - 1) / tpb;

    int num_steps = 200000;

    kernel_make_k2<<<grid_k, block2d>>>(k2, Nx, Ny);
    init_random_states<<<nb, tpb>>>(random_states, 42, Ntot);

    std::vector<float>  phi_host(Ntot);
    std::ofstream fout("phi_final.bin", std::ios::binary);
    std::ofstream fout_probe("probe.txt");
    for(int step = 0; step < num_steps; step++){
        accumulate_probe_forces<<<grid_float, block2d>>>(phi, probe, probe_force, Nx, Ny, interaction_strength, probe_radius);
        //update noise
        fill_noise<<<nb, tpb>>>(random_states,noise, Ntot);
        //compute forces from the probe to the field
        compute_spatial_forces<<<grid_float, block2d>>>(phi, probe, spatial_forces, Nx, Ny, interaction_strength, probe_radius);
        //get fft of noise, spatial forces, and field
        cufftExecR2C(plan_r2c, noise, noise_k);
        cufftExecR2C(plan_r2c, spatial_forces, spatial_forces_k);
        cufftExecR2C(plan_r2c, phi, phi_k);
        //update field in Fourier space
        kernel_update_modes<<<grid_k, block2d>>>(phi_k, spatial_forces_k, noise_k, k2, phi_k_next, Nx, Ny, mu, kappa, mobility, temp, dt);
        //transform back to real space and normalize
        cufftExecC2R(plan_c2r, phi_k_next, phi);
        kernel_fft_normalize<<<nb, tpb>>>(phi, Ntot, invN);
        probe_step<<<1,1>>>(probe, probe_force, u, well_strength, well_velocity, well_y, dt, step*dt, Nx, Ny);
        //record
        float2 probe_position;
        cudaMemcpy(&probe_position, probe, sizeof(float2), cudaMemcpyDeviceToHost);
        float u_host;
        cudaMemcpy(&u_host, u, sizeof(float), cudaMemcpyDeviceToHost);
        fout_probe << probe_position.x << " " << probe_position.y << " " << u_host << "\n";
        fout_probe.flush();
        if(step % 100 == 0)
        {
            std::cout << "Step " << step << " Probe position: (" << probe_position.x << ", " << probe_position.y << ")\n";
            cudaMemcpy(phi_host.data(), phi, Ntot * sizeof(float), cudaMemcpyDeviceToHost);
            fout.write(reinterpret_cast<const char*>(phi_host.data()), Ntot * sizeof(float));
        }
    }
    fout_probe.close();
    fout.close();
}