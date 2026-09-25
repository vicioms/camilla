#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

const int NUM_THREADS = 256;
constexpr double PI = 3.14159265358979323846;
constexpr double EPSILON = 1e-7;


// ============================================================
// Parameters
// ============================================================

struct simulationParams
{
    // field
    double mu;
    double r;
    double mobility;

    // particle
    double v_x;       // trap x velocity
    double v_y;       // trap y velocity
    double kappa;     // trap stiffness
    double gamma0;    // friction coefficient
    double a;         // particle size

    // interaction
    double lambda;

    bool conserved_field;
};


// ============================================================
// Simulation
// ============================================================

struct simulation
{
    // generic
    int Nx, Ny, Nyh;
    double dx, dy, Lx, Ly, dt;

    // physical parameters
    simulationParams params;

    // FFT plans
    cufftHandle fft_plan;
    cufftHandle ifft_plan;

    // host particle variables
    double z_x, z_y;
    double new_z_x, new_z_y;

    // device Fourier quantities
    double *q_x;
    double *q_y;
    double *q2;
    double *V_q;
    double *integration_factor_q;
    double *phi1_integration_factor_q;

    // field
    double *phi;
    cufftDoubleComplex *phi_q;
    cufftDoubleComplex *new_phi_q;

    // particle force [Fx, Fy]
    double *z_force;
};


// ============================================================
// Initialization
// ============================================================

void init_simulation(simulation& sim, int Nx, int Ny, double dx, double dy, double dt, simulationParams params)
{
    sim.Nx = Nx;
    sim.Ny = Ny;
    sim.Nyh = Ny / 2 + 1;

    sim.dx = dx;
    sim.dy = dy;
    sim.Lx = Nx * dx;
    sim.Ly = Ny * dy;
    sim.dt = dt;
    sim.params = params;

    sim.z_x = 0.0;
    sim.z_y = 0.0;
    sim.new_z_x = 0.0;
    sim.new_z_y = 0.0;

    const int Nq = sim.Nx * sim.Nyh;

    // --------------------------------------------------------
    // Build q-dependent quantities on host
    // --------------------------------------------------------

    double *q_x = new double[sim.Nx];
    double *q_y = new double[sim.Nyh];
    double *q2 = new double[Nq];
    double *Vq = new double[Nq];
    double *integration_factor_q = new double[Nq];
    double *phi1_integration_factor_q = new double[Nq];

    for (int i = 0; i < sim.Nx; ++i)
    {
        int ni = (i <= sim.Nx / 2) ? i : i - sim.Nx;
        q_x[i] = 2.0 * PI * ni / sim.Lx;
    }

    // D2Z stores only the non-negative half of the last dimension
    for (int j = 0; j < sim.Nyh; ++j) q_y[j] = 2.0 * PI * j / sim.Ly;

    for (int i = 0; i < sim.Nx; ++i)
    {
        for (int j = 0; j < sim.Nyh; ++j)
        {
            int idx = i * sim.Nyh + j;

            q2[idx] = q_x[i] * q_x[i] + q_y[j] * q_y[j];

            // Gaussian Fourier envelope only:
            // V_q = exp(-a^2 q^2 / 2)
            // lambda is NOT included here
            Vq[idx] = exp(-0.5 * params.a * params.a * q2[idx]);

            // Non-conserved: g_q = M (r + mu q^2)
            // Conserved:     g_q = M q^2 (r + mu q^2)
            double g_q = params.mobility * (params.r + params.mu * q2[idx]);
            if (params.conserved_field) g_q *= q2[idx];

            double factor_q = g_q * sim.dt;

            // E_q = exp(-g_q dt)
            integration_factor_q[idx] = exp(-factor_q);

            // phi_1(x) = (1-exp(-x))/x
            phi1_integration_factor_q[idx] =
                (fabs(factor_q) < EPSILON) ? 1.0 : -expm1(-factor_q) / factor_q;
        }
    }

    // --------------------------------------------------------
    // Allocate q-dependent arrays on GPU
    // --------------------------------------------------------

    cudaMalloc(&sim.q_x, sim.Nx * sizeof(double));
    cudaMalloc(&sim.q_y, sim.Nyh * sizeof(double));
    cudaMalloc(&sim.q2, Nq * sizeof(double));
    cudaMalloc(&sim.V_q, Nq * sizeof(double));
    cudaMalloc(&sim.integration_factor_q, Nq * sizeof(double));
    cudaMalloc(&sim.phi1_integration_factor_q, Nq * sizeof(double));

    cudaMemcpy(sim.q_x, q_x, sim.Nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sim.q_y, q_y, sim.Nyh * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sim.q2, q2, Nq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sim.V_q, Vq, Nq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sim.integration_factor_q, integration_factor_q, Nq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(sim.phi1_integration_factor_q, phi1_integration_factor_q, Nq * sizeof(double), cudaMemcpyHostToDevice);

    delete[] q_x;
    delete[] q_y;
    delete[] q2;
    delete[] Vq;
    delete[] integration_factor_q;
    delete[] phi1_integration_factor_q;

    // --------------------------------------------------------
    // Fields
    // --------------------------------------------------------

    cudaMalloc(&sim.phi, sim.Nx * sim.Ny * sizeof(double));
    cudaMalloc(&sim.phi_q, Nq * sizeof(cufftDoubleComplex));
    cudaMalloc(&sim.new_phi_q, Nq * sizeof(cufftDoubleComplex));
    cudaMalloc(&sim.delta_phi_q, Nq * sizeof(cufftDoubleComplex));

    // final particle force [Fx, Fy]
    cudaMalloc(&sim.z_force, 2 * sizeof(double));

    // --------------------------------------------------------
    // FFT plans
    // --------------------------------------------------------

    cufftPlan2d(&sim.fft_plan, sim.Nx, sim.Ny, CUFFT_D2Z);
    cufftPlan2d(&sim.ifft_plan, sim.Nx, sim.Ny, CUFFT_Z2D);
}


// ============================================================
// FFT
// ============================================================

__global__ void scale(double* data, int size, double scale_factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] *= scale_factor;
}


void fft_phi(simulation& sim)
{
    cufftExecD2Z(sim.fft_plan, sim.phi, sim.phi_q);
}


void ifft_phi(simulation& sim)
{
    cufftExecZ2D(sim.ifft_plan, sim.phi_q, sim.phi);

    int N = sim.Nx * sim.Ny;
    int n_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

    scale<<<n_blocks, NUM_THREADS>>>(sim.phi, N, 1.0 / static_cast<double>(N));
}


// ============================================================
// Force on particle
// ============================================================

__global__ void force_on_particle(
    const cufftDoubleComplex* phi_q,
    const double* V_q,
    const double* q_x,
    const double* q_y,
    double z_x, double z_y,
    double* force,
    int Nx, int Ny, int Nyh)
{
    // Real-space force:
    //
    // F(Z) = - ∫ dx phi(x) grad_Z V(x-Z)
    //
    // Since grad_Z V(x-Z) = -grad_x V(x-Z):
    //
    // F(Z) = ∫ dx phi(x) grad_x V(x-Z)
    //
    // With:
    //
    // V(x-Z) = ∫ Dq V(q) exp(i q·(x-Z))
    //
    // we obtain:
    //
    // F(Z) = ∫ Dq (-i q) V(q) exp(i q·Z) phi(q)
    //
    // for an even potential V(-q) = V(q).
    //
    // Therefore:
    //
    // F_q = q V(q) Im[phi(q) exp(i q·Z)]
    //
    // V_q here contains only exp(-a^2 q^2 / 2).
    // The factor -lambda is applied afterwards.

    extern __shared__ double2 shared_force[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nq = Nx * Nyh;

    double2 local_force = make_double2(0.0, 0.0);

    if (idx < Nq)
    {
        int i = idx / Nyh;
        int j = idx % Nyh;

        // q · Z
        double overlap = q_x[i] * z_x + q_y[j] * z_y;

        double sin_overlap, cos_overlap;
        sincos(overlap, &sin_overlap, &cos_overlap);

        // Im[phi(q) exp(i q·Z)]
        double imag_part = phi_q[idx].x * sin_overlap + phi_q[idx].y * cos_overlap;

        // D2Z stores only q_y >= 0.
        //
        // Interior positive-q_y modes represent the missing negative-q_y
        // partner and therefore receive weight 2.
        //
        // q_y = 0 is unique.
        // For even Ny, q_y = Ny/2 is also unique.
        double weight = 1.0;
        if (j > 0 && !(Ny % 2 == 0 && j == Ny / 2)) weight = 2.0;

        // Raw cuFFT forward coefficients are unnormalized:
        //
        // F = (1 / Nx Ny) sum_q F_q
        double prefactor = weight * V_q[idx] * imag_part / static_cast<double>(Nx * Ny);

        local_force.x = prefactor * q_x[i];
        local_force.y = prefactor * q_y[j];
    }

    // --------------------------------------------------------
    // Block reduction
    // --------------------------------------------------------

    shared_force[tid] = local_force;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_force[tid].x += shared_force[tid + stride].x;
            shared_force[tid].y += shared_force[tid + stride].y;
        }

        __syncthreads();
    }

    // One atomic addition per component per block
    if (tid == 0)
    {
        atomicAdd(&force[0], shared_force[0].x);
        atomicAdd(&force[1], shared_force[0].y);
    }
}


// ============================================================
// Particle update
// ============================================================

void compute_new_z(simulation& sim, double t)
{
    int Nq = sim.Nx * sim.Nyh;
    int n_blocks = (Nq + NUM_THREADS - 1) / NUM_THREADS;

    // Reset Fx, Fy
    cudaMemset(sim.z_force, 0, 2 * sizeof(double));

    force_on_particle<<<n_blocks, NUM_THREADS, NUM_THREADS * sizeof(double2)>>>(
        sim.phi_q, sim.V_q, sim.q_x, sim.q_y,
        sim.z_x, sim.z_y, sim.z_force,
        sim.Nx, sim.Ny, sim.Nyh
    );

    // Copy only the final Fx, Fy to CPU
    double force[2];
    cudaMemcpy(force, sim.z_force, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    // Physical interaction is -lambda V
    double F_int_x = -sim.params.lambda * force[0];
    double F_int_y = -sim.params.lambda * force[1];

    
}


// ============================================================
// Force on field
// ============================================================ 

__global__ compute_new_phi_q(
    cufftDoubleComplex* phi_q,
    cufftDoubleComplex* new_phi_q,
    double z_x, 
    double z_y,
    double lambda,
    const double* q_x,
    const double* q_y,
    const double* q2,
    const double* V_q,
    const double* integration_factor_q,
    const double* phi1_integration_factor_q,
    const double* delta_phi_q,
    bool conserved_field,
    int Nx, int Nyh)
{
    // potential is - lambda \int dx \phi(x) V(x-Z)
    // gradient is - lambda \phi(x) V(x-Z)
    // the force is:
    // (-1)^a \nabla^{2a} lambda V(x-Z)
    // a = 0 for non-conserved, a = 1 for conserved
    // for a = 0 we have:
    // \lambda V(x-Z) 
    // taking FT: \lambda \int dx  V(x-Z) exp(-i q x)
    // = \lambda \int dy V(y) exp(-i q (y+Z))
    // = \lambda exp(-i q Z) \int dy V(y) exp(-i q y)
    // = \lambda exp(-i q Z) V(q)
    // for a = 1 we have:
    // \lambda \nabla^2 V(x-Z)
    // taking FT: \lambda \int dx  \nabla^2 V(x-Z) exp(-i q x)
    // = - \lambda q^2 \int dx V(x-Z) exp(-i q x)
    // = - \lambda q^2 exp(-i q Z) V(q)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nq = Nx * Nyh;
    

    if (idx < Nq)
    {
        new_phi_q[idx].x = integration_factor_q[idx] * phi_q[idx].x;
        new_phi_q[idx].y = integration_factor_q[idx] * phi_q[idx].y;
        
    }
}



// ============================================================
// Cleanup
// ============================================================

void free_simulation(simulation& sim)
{
    cufftDestroy(sim.fft_plan);
    cufftDestroy(sim.ifft_plan);

    cudaFree(sim.q_x);
    cudaFree(sim.q_y);
    cudaFree(sim.q2);
    cudaFree(sim.V_q);
    cudaFree(sim.integration_factor_q);
    cudaFree(sim.phi1_integration_factor_q);

    cudaFree(sim.phi);
    cudaFree(sim.phi_q);
    cudaFree(sim.delta_phi_q);

    cudaFree(sim.z_force);
}