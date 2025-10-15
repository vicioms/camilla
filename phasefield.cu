#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>

using real = float;
using cplx = cufftComplex;

__host__ __device__ __forceinline__ int idx2(int x, int y, int Nx) { return y * Nx + x; }

// ---- f'(c) = c^3 - c in real space
__global__ void kernel_fprime(const real* __restrict__ c, real* __restrict__ fp, int Nx, int Ny){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    int id = idx2(ix, iy, Nx);
    real v = c[id];
    fp[id] = v * (v * v - 1.0f);
}

// ---- precompute k^2 on the R2C grid (Nxk = Nx/2+1 by Ny)
__global__ void kernel_make_k2(real* __restrict__ k2, int Nx, int Ny){
    int Nxk = Nx/2 + 1;
    int kx = blockIdx.x * blockDim.x + threadIdx.x;   // 0..Nx/2
    int ky = blockIdx.y * blockDim.y + threadIdx.y;   // 0..Ny-1
    if (kx >= Nxk || ky >= Ny) return;
    int id = idx2(kx, ky, Nxk);
    int kx_phys = kx;                                  // non-negative
    int ky_phys = (ky <= Ny/2) ? ky : ky - Ny;         // wrap negative freqs
    real k2v = real(kx_phys * kx_phys + ky_phys * ky_phys);
    k2[id] = k2v;
}

// ---- semi-implicit update in Fourier space
// ck_next = (ck - dt*M*k2*fp_k) / (1 + dt*M*kappa*k2*k2)
__global__ void kernel_update_modes(const cplx* __restrict__ ck,
                                    const cplx* __restrict__ fp_k,
                                    const real* __restrict__ k2,
                                    cplx* __restrict__ ck_next,
                                    int Nx, int Ny, real M, real kappa, real dt){
    int Nxk = Nx/2 + 1;
    int kx = blockIdx.x * blockDim.x + threadIdx.x;
    int ky = blockIdx.y * blockDim.y + threadIdx.y;
    if (kx >= Nxk || ky >= Ny) return;
    int id = idx2(kx, ky, Nxk);

    real k2v = k2[id];
    real denom = 1.0f + dt * M * kappa * k2v * k2v;

    cplx C = ck[id];
    cplx F = fp_k[id];

    cplx num;
    real a = dt * M * k2v;
    num.x = C.x - a * F.x;
    num.y = C.y - a * F.y;

    ck_next[id].x = num.x / denom;
    ck_next[id].y = num.y / denom;
    // (k=0) mode stays unchanged automatically because k2=0 -> num=C, denom=1
}

// ---- normalize after inverse FFT (cuFFT C2R is unnormalized)
__global__ void kernel_normalize(real* __restrict__ c, int Ntot, real invN){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Ntot) c[i] *= invN;
}

// ---- simple +/-1 random init (binary spinodal-like)
void random_initialize(real* d_c, int Nx, int Ny){
    std::mt19937 gen(42);
    std::uniform_real_distribution<real> unif(-1.0f, 1.0f);
    std::vector<real> h(Nx * Ny);
    for (int i = 0; i < Nx * Ny; ++i) h[i] = (unif(gen) > 0.0f) ? 1.0f : -1.0f;
    cudaMemcpy(d_c, h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice);
}

int main(){
    // --- grid
    const int Nx = 2048;                 // tune
    const int Ny = 2048;
    const int Ntot = Nx * Ny;
    const int Nxk = Nx/2 + 1;
    const size_t Nk = size_t(Ny) * size_t(Nxk); // (Ny, Nx/2+1)

    // --- physics
    const real dt   = 1e-3f;
    const real M    = 1.0f;              // mobility
    const real kappa = 0.1f;              // gradient energy coeff
    const real invN = 1.0f / real(Nx * Ny);

    // --- device arrays
    real *d_c, *d_fp;
    cudaMalloc(&d_c,  Ntot * sizeof(real));
    cudaMalloc(&d_fp, Ntot * sizeof(real));

    cplx *d_ck, *d_fpk, *d_ck_next;
    cudaMalloc(&d_ck,      Nk * sizeof(cplx));
    cudaMalloc(&d_fpk,     Nk * sizeof(cplx));
    cudaMalloc(&d_ck_next, Nk * sizeof(cplx));

    real *d_k2;
    cudaMalloc(&d_k2, Nk * sizeof(real));

    // --- init
    random_initialize(d_c, Nx, Ny);

    // --- cuFFT plans (row-major; sizes are (Ny, Nx))
    cufftHandle plan_r2c, plan_c2r;
    cufftPlan2d(&plan_r2c, Ny, Nx, CUFFT_R2C);
    cufftPlan2d(&plan_c2r, Ny, Nx, CUFFT_C2R);

    // --- launch params
    dim3 block2d(16, 16);
    dim3 grid_real((Nx + block2d.x - 1)/block2d.x,
                   (Ny + block2d.y - 1)/block2d.y);
    dim3 grid_k((Nxk + block2d.x - 1)/block2d.x,
                (Ny  + block2d.y - 1)/block2d.y);
    int tpb = 256;
    int nb  = (Ntot + tpb - 1) / tpb;

    // --- precompute k^2 once
    kernel_make_k2<<<grid_k, block2d>>>(d_k2, Nx, Ny);

    // --- time loop
    const int steps = 100000;
    std::vector<real> h_c(Ntot);
    std::ofstream fout("phi_final.bin", std::ios::binary);
    for (int s = 0; s < steps; ++s){
        // 1) f'(c) in real space
        kernel_fprime<<<grid_real, block2d>>>(d_c, d_fp, Nx, Ny);

        // 2) FFT c and f'(c)
        cufftExecR2C(plan_r2c, d_c,  d_ck);
        cufftExecR2C(plan_r2c, d_fp, d_fpk);

        // 3) semi-implicit update in k-space
        kernel_update_modes<<<grid_k, block2d>>>(d_ck, d_fpk, d_k2, d_ck_next, Nx, Ny, M, kappa, dt);

        // 4) inverse FFT to real space (unnormalized)
        cufftExecC2R(plan_c2r, d_ck_next, d_c);

        // 5) normalize
        kernel_normalize<<<nb, tpb>>>(d_c, Ntot, invN);

        if(s % 1000 == 0)
        {
            cudaMemcpy(h_c.data(), d_c, Ntot * sizeof(real), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            printf("Step %d / %d\n", s, steps);
            fout.write(reinterpret_cast<const char*>(h_c.data()), Ntot * sizeof(real));
        }
        // (optional) diagnostics every N steps…
        // if ((s & 255) == 0) { /* copy small stats to host, print */ }
    }

    fout.close();

    // --- cleanup
    cufftDestroy(plan_r2c);
    cufftDestroy(plan_c2r);
    cudaFree(d_k2);
    cudaFree(d_ck_next); cudaFree(d_fpk); cudaFree(d_ck);
    cudaFree(d_fp); cudaFree(d_c);
    return 0;
}