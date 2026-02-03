#pragma once
#include "../helper_matrices.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <random>
using namespace std;



inline __device__ __host__ int pmod(int i, int n)
{
    return (i % n + n) % n;
};

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


struct CubWorkspace
{
    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CubWorkspace() = default;

    void ensure_size(size_t required_bytes)
    {
        if(temp_storage == nullptr || temp_storage_bytes < required_bytes) {
            if(temp_storage != nullptr) {
                cudaFree(temp_storage);
            }
            cudaMalloc(&temp_storage, required_bytes);
            temp_storage_bytes = required_bytes;
        }
    };

    void free()
    {
        if(temp_storage != nullptr) {
            cudaFree(temp_storage);
            temp_storage = nullptr;
            temp_storage_bytes = 0;
        }
    };

    ~CubWorkspace()
    {
        free();
    };
};

__device__ __host__ __forceinline__ void shape_self_interaction_forces(smat2 shape, float k, float g, float k_el, float a0, smat2& shape_force)
{
    float shape_det = det(shape);
    float shape_det_inv = 1.0f / shape_det;

    smat2 shape_inv = make_smat2( shape.yy*shape_det_inv,
                                        -shape.xy*shape_det_inv,  shape.xx*shape_det_inv);

    float trace_inv = shape_inv.xx + shape_inv.yy;
    if(k != 0.0f | g != 0.0f)
    {
        shape_force += sqrtf(shape_det)* (k*square(shape_inv) - (0.5f*(k*trace_inv + g)*shape_inv));
    };
    if(k_el != 0.0f)
    {
        shape_force += (k_el*(a0 - shape_det)* sqrtf(shape_det))*shape_inv;
    } 
};

__device__ __host__ __forceinline__ void shape_polarity_self_interaction_forces(smat2 shape, float theta, float k_align, float anisotropy_param, smat2& shape_force, float& theta_force)
{
   
};

__device__ __host__ __forceinline__ void shape_interaction_forces(float2 r_vec,
                    smat2 shape, smat2 shape_other, float interaction_strength, float2& r_force, smat2& shape_force)
{
    smat2 shape_sum = shape + shape_other;
    smat2 shape_inv = inverse(shape);
    float shape_sum_det = det(shape_sum);
    float shape_sum_inv_det = 1.0f / shape_sum_det;

    smat2 shape_sum_inv = make_smat2( shape_sum.yy*shape_sum_inv_det,
                                        -shape_sum.xy*shape_sum_inv_det,  shape_sum.xx*shape_sum_inv_det);
    float2 r_tilde = matmul(shape_sum_inv, r_vec);
    float r_tilde2 = dot(r_tilde, r_tilde);
    float exp_term = expf(-0.5f * r_tilde2);
    float shape_det = det(shape);
    float shape_other_det = det(shape_other);
    float potential = interaction_strength * exp_term*sqrtf(shape_det*shape_other_det*shape_sum_inv_det);
    r_force += potential * r_tilde;
    shape_force += (0.5f*potential)*(shape_sum_inv - shape_inv - outer(r_tilde));
};


struct Particle{
    float2 position;
    float theta;
    smat2 shape;
    float2 position_force;
    float omega;
    smat2 shape_force;
};

__global__ void particleSimulation_compute_pcell_ids(
    const Particle* __restrict__ particles,
    int* pcell_ids,
    float2 box_origin,
    float cell_size,
    int2 grid_size,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float2 pos = particles[idx].position;
    int2 cell;
    cell.x = (int)floorf((pos.x - box_origin.x) / cell_size);
    cell.y = (int)floorf((pos.y - box_origin.y) / cell_size);
    // apply periodic boundary conditions
    cell.x = pmod(cell.x, grid_size.x);
    cell.y = pmod(cell.y, grid_size.y);

    int pcell_id = cell.x + cell.y * grid_size.x;
    pcell_ids[idx] = pcell_id;
};

__global__ void scatter_start_end(
    const int* __restrict__ indices,
    const int* __restrict__ offsets,
    const int* __restrict__ counts,
    int* start,
    int* end,
    int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    int idx = indices[r];
    int offset = offsets[r];
    int count = counts[r];
    start[idx] = offset;
    end[idx] = offset + count;
};


//beware that not all cells are occupied
//length of cell_ids, cell_offsets, cell_counts is num_occupied_cells
//length of cell_start, cell_end is num_cells
//we loop one block per cell
__global__ void particleSimulation_loop(
                                const int* __restrict__ cell_ids,
                                const int* __restrict__ cell_start,
                                const int* __restrict__ cell_end,
                                int num_occupied_cells,
                                float2 box_origin,
                                float2 box,
                                int2 grid_size,
                                float cutoff_radius,
                                float v0,
                                float shape_force_strength,
                                Particle* particles,
                                int num_particles)
{
    int r = blockIdx.x;
    if(r >= num_occupied_cells) return;

    int home_cell_id = cell_ids[r];
    int home_start =  cell_start[home_cell_id];
    if(home_start < 0 || home_start >= num_particles) return;
    int home_end = cell_end[home_cell_id];
    int2 home_cell = make_int2(0,0);
    home_cell.x = home_cell_id % grid_size.x;
    home_cell.y = home_cell_id / grid_size.x;


    float cutoff_radius2 = cutoff_radius * cutoff_radius;

    for(int i = threadIdx.x + home_start; i < home_end; i += blockDim.x)
    {
       Particle particle_i = particles[i];
       float2 pos_i = particle_i.position;
       float theta = particle_i.theta;
       smat2 shape_i = particle_i.shape;
       float2 pos_force = make_float2(0.0f);
       float omega = 0.0f;
       smat2 shape_force = make_smat2(0.0f);
       shape_self_interaction_forces(shape_i, 1.0f, 0.0f, 2.0f, cutoff_radius2 / 25.0f  , shape_force);

       //loop over neighboring cells
        for(int dy = -1; dy <= 1; dy++)
        for(int dx = -1; dx <= 1; dx++)
        {
            int2 neighbor_cell;
            neighbor_cell.x = pmod(home_cell.x + dx, grid_size.x);
            neighbor_cell.y = pmod(home_cell.y + dy, grid_size.y);
            int neighbor_cell_id = neighbor_cell.x + neighbor_cell.y * grid_size.x;
            int neighbor_start = cell_start[neighbor_cell_id];
            if(neighbor_start == -1) continue; //empty cell
            int neighbor_end = cell_end[neighbor_cell_id];
            for(int j = neighbor_start; j < neighbor_end; j++)
            {
                if(j == i) continue; //skip self
                Particle particle_j = particles[j];
                float2 pos_j = particle_j.position;
                float2 r_vec = wrapped_float2_diff(pos_i, pos_j, box);
                float r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y;
                if(r2 > cutoff_radius2) continue;
                shape_interaction_forces(r_vec, shape_i, particle_j.shape, shape_force_strength, pos_force, shape_force);
            }
        }
        particles[i].position_force = pos_force + v0 * make_float2(cosf(theta), sinf(theta));
        particles[i].omega = omega;
        particles[i].shape_force = shape_force;
    }
};


__global__ void particleSimulation_step(
    Particle* particles,
    curandStatePhilox4_32_10_t* random_states,
    int num_particles,
    float2 box_origin, float2 box,
    float eta,
    float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Update positions and polarities based on forces
    float2 position = particles[idx].position;
    position += particles[idx].position_force * dt;
    //if(eta > 0.0f) {
    //    curandStatePhilox4_32_10_t local_state = random_states[idx];
    //    float noise_x = curand_normal(&local_state) * sqrtf(2.0f * eta * dt);
    //    float noise_y = curand_normal(&local_state) * sqrtf(2.0f * eta * dt);
    //    float noise_z = curand_normal(&local_state) * sqrtf(2.0f * eta * dt);
    //    position.x += noise_x;
    //    position.y += noise_y;
    //    position.z += noise_z;
    //    random_states[idx] = local_state;
    //};
    position = wrap_float2(position, box_origin, box);
    particles[idx].position = position;
    float theta = particles[idx].theta;
    theta += particles[idx].omega * dt;
    if(eta > 0.0f) {
        curandStatePhilox4_32_10_t local_state = random_states[idx];
        float rotational_noise = curand_normal(&local_state) * sqrtf(2.0f * eta * dt);
        theta += rotational_noise;
        random_states[idx] = local_state;
    };
    particles[idx].theta = theta - 2.0f * M_PI * floorf(theta / (2.0f * M_PI));
    particles[idx].shape += particles[idx].shape_force * dt;
    // Reset forces for the next iteration
    particles[idx].position_force = make_float2(0.0f);
    particles[idx].omega = 0.0f;
    particles[idx].shape_force = make_smat2(0.0f);
};


__global__ void init_random_states(curandStatePhilox4_32_10_t* states,
                            unsigned long long seed, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        // sequence = gid makes streams independent per thread
        curand_init(seed, /*sequence*/ gid, /*offset*/ 0, &states[gid]);
    }
};

class ParticleSimulation {
public:
    //system parameters
    float2 box_origin;
    float2 box;
    float cell_size;
    int2 grid_size;
    int num_cells;
    float skin;
    float cutoff_radius;
    float eta;
    //cuda parameters
    int num_threads_per_block = 64;
    //simulation variables
    Particle* particles = nullptr;
    Particle* temp_particles = nullptr;
    float2* position_before_update = nullptr;
    float* displacements_from_update = nullptr;
    int num_particles = 0;
    int* pcell_ids = nullptr;
    int* sorted_pcell_ids = nullptr;
    int* cell_ids = nullptr;
    int* cell_counts = nullptr; 
    int* cell_offsets = nullptr;
    int* cell_start = nullptr;
    int* cell_end = nullptr;
    int num_occupied_cells = 0;
    //top_k<K>* knns = nullptr;
    curandStatePhilox4_32_10_t* random_states = nullptr;
    CubWorkspace sort_workspace;
    CubWorkspace rle_workspace;
    CubWorkspace scan_workspace;
    CubWorkspace max_workspace;
    //potential params
    float v0;
    float force_strength;

    ParticleSimulation(int num_particles, float v0, float f_strength, float2 box_origin, float2 box, float cell_size, float cutoff_radius, float eta, int seed)
    {
        this->v0 = v0;
        this->force_strength = f_strength;
        this->box_origin = box_origin;
        this->box = box;
        this->cell_size = cell_size;
        this->cutoff_radius = cutoff_radius;
        this->skin = fmaxf(cell_size - cutoff_radius, 0.0f);
        this->eta = eta;
         //initialize random states
        grid_size.x = (int)ceilf((box.x - box_origin.x) / cell_size);
        grid_size.y = (int)ceilf((box.y - box_origin.y) / cell_size);
        num_cells = grid_size.x * grid_size.y;
        this->num_particles = num_particles;
        allocate(seed);
    };

    void allocate(unsigned long long seed)
    {
        cudaMalloc(&particles, num_particles * sizeof(Particle));
        cudaMalloc(&temp_particles, num_particles * sizeof(Particle));
        cudaMalloc(&pcell_ids, num_particles * sizeof(int));
        cudaMalloc(&sorted_pcell_ids, num_particles * sizeof(int));
        cudaMalloc(&cell_ids, num_particles * sizeof(int));
        cudaMalloc(&cell_counts, num_particles * sizeof(int));
        cudaMalloc(&cell_offsets, num_particles * sizeof(int));
        cudaMalloc(&cell_start, num_cells * sizeof(int));
        cudaMalloc(&cell_end, num_cells * sizeof(int));
        cudaMalloc(&random_states, num_particles * sizeof(curandStatePhilox4_32_10_t));
        int num_blocks = (num_particles + num_threads_per_block - 1) / num_threads_per_block;
        init_random_states<<<num_blocks, num_threads_per_block>>>(random_states, seed, num_particles);
        cudaDeviceSynchronize();
        //knns = new top_k<K>[num_particles];
        //cudaMalloc(&position_before_update, num_particles * sizeof(float4));
        //cudaMalloc(&displacements_from_update, num_particles * sizeof(float));
    };

    void random_init(default_random_engine& rng, float2 min_pos, float2 max_pos)
    {
        uniform_real_distribution<float> dist_x(max(min_pos.x, box_origin.x), min(max_pos.x, box_origin.x + box.x));
        uniform_real_distribution<float> dist_y(max(min_pos.y, box_origin.y), min(max_pos.y, box_origin.y + box.y));
        uniform_real_distribution<float> dist_theta(0.0f, 2.0f * M_PI);
        Particle* h_particles = new Particle[num_particles];
        for(int i = 0; i < num_particles; i++)
        {
            h_particles[i].position = make_float2(dist_x(rng), dist_y(rng));
            h_particles[i].theta = dist_theta(rng);
            h_particles[i].shape = make_smat2( cutoff_radius*cutoff_radius/25.0f, 0.0f, cutoff_radius*cutoff_radius/25.0f);
            h_particles[i].position_force = make_float2(0.0f);
            h_particles[i].omega = 0.0f;
            h_particles[i].shape_force = make_smat2(0.0f);
        }
        cudaMemcpy(particles, h_particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);
        delete[] h_particles;
    };

    int updateCellList()
    {
        //compute pcell ids
        int num_blocks = (num_particles + num_threads_per_block - 1) / num_threads_per_block;
        particleSimulation_compute_pcell_ids<<<num_blocks, num_threads_per_block>>>(
            particles,
            pcell_ids,
            box_origin,
            cell_size,
            grid_size,
            num_particles);
        //sort particles by pcell ids
        size_t sort_required_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr,
            sort_required_bytes, 
            (int*)nullptr, 
            (int*)nullptr, 
            (Particle*)nullptr, 
            (Particle*)nullptr, 
            num_particles);

        sort_workspace.ensure_size(sort_required_bytes);
        cub::DeviceRadixSort::SortPairs(sort_workspace.temp_storage,
            sort_required_bytes,
            pcell_ids,
            sorted_pcell_ids,
            particles,
            temp_particles,
            num_particles);

        //swap particle pointers
        Particle* tmp = particles;
        particles = temp_particles;
        temp_particles = tmp;
        
        //run length encode to find uniquely occupied cells, their counts and offsets
        size_t rle_required_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr,
            rle_required_bytes,
            (int*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            (int*)nullptr,
            num_particles);
        rle_workspace.ensure_size(rle_required_bytes);

        //static int* d_num_runs_out = nullptr;
        //static bool first_run = true;
        //if(first_run) {
        //    cudaMalloc(&d_num_runs_out, sizeof(int));
        //    first_run = false;
        //};

        int* d_num_runs_out = nullptr;
        cudaMalloc(&d_num_runs_out, sizeof(int));

        cub::DeviceRunLengthEncode::Encode(rle_workspace.temp_storage,
            rle_required_bytes,
            sorted_pcell_ids,
            cell_ids,
            cell_counts,
            d_num_runs_out,
            num_particles);

        int h_num_runs = 0;
        cudaMemcpy(&h_num_runs, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);
        
        //exclusive scan to get cell offsets
        size_t scan_required_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr,
            scan_required_bytes,
            (int*)nullptr,
            (int*)nullptr,
            h_num_runs);
        scan_workspace.ensure_size(scan_required_bytes);
        cub::DeviceScan::ExclusiveSum(scan_workspace.temp_storage,
            scan_workspace.temp_storage_bytes,
            cell_counts,
            cell_offsets,
            h_num_runs);

        cudaMemset(cell_start, 0xFF, num_cells * sizeof(int));
        cudaMemset(cell_end, 0xFF, num_cells * sizeof(int));

        int num_blocks_scatter = (h_num_runs + num_threads_per_block - 1) / num_threads_per_block;
        scatter_start_end<<<num_blocks_scatter, num_threads_per_block>>>(
            cell_ids,
            cell_offsets,
            cell_counts,
            cell_start,
            cell_end,
            h_num_runs);
        num_occupied_cells = h_num_runs;
        //printf("Number of occupied cells: %d\n", num_occupied_cells);
        return h_num_runs;
    };


    void loop()
    {
        particleSimulation_loop<<<num_occupied_cells, num_threads_per_block>>>(
            cell_ids,
            cell_start,
            cell_end,
            num_occupied_cells,
            box_origin,
            box,
            grid_size,
            cutoff_radius,
            v0,
            force_strength,
            particles,
            num_particles);
    };
    
    void step(float dt)
    {
        int num_blocks = (num_particles + num_threads_per_block - 1) / num_threads_per_block;
        particleSimulation_step<<<num_blocks, num_threads_per_block>>>(
            particles,
            random_states,
            num_particles,
            box_origin, box,
            eta,
            dt);
        
    };

    void free()
    {
        if(particles != nullptr) {
            cudaFree(particles);
            particles = nullptr;
        };
        if(temp_particles != nullptr) {
            cudaFree(temp_particles);
            temp_particles = nullptr;
        };
        if(pcell_ids != nullptr) {
            cudaFree(pcell_ids);
            pcell_ids = nullptr;
        };
        if(sorted_pcell_ids != nullptr) {
            cudaFree(sorted_pcell_ids);
            sorted_pcell_ids = nullptr;
        };
        if(cell_ids != nullptr) {
            cudaFree(cell_ids);
            cell_ids = nullptr;
        };
        if(cell_counts != nullptr) {
            cudaFree(cell_counts);
            cell_counts = nullptr;
        };
        if(cell_offsets != nullptr) {
            cudaFree(cell_offsets);
            cell_offsets = nullptr;
        };
        if(cell_start != nullptr) {
            cudaFree(cell_start);
            cell_start = nullptr;
        };
        if(cell_end != nullptr) {
            cudaFree(cell_end);
            cell_end = nullptr;
        };
        sort_workspace.free();
        rle_workspace.free();
        scan_workspace.free();
    };

    int set_threads_per_block(int n)
    {
        if(n <= 0) return num_threads_per_block;
        num_threads_per_block = n;
        return num_threads_per_block;
    };
};