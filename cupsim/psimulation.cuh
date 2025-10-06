#pragma once
#include "helper_math.h"
#include <random>
#include <cub/cub.cuh>
#include "cuda.h"
#include "cuda_runtime.h"
using namespace std;

#ifndef TILE_DIM
#define TILE_DIM 64
#endif

template<int K>
struct top_k
{
    float values[K];
    int indices[K];
    int max_pos;
    int num_occupied;

    __device__ void init(float inf=1e12f) {
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

inline __device__ __host__ int pmod(int i, int n)
{
    return (i % n + n) % n;
};

__host__ __device__ __forceinline__
float4 wrapped_float4_diff(float4 a, float4 b, float3 box)
{
    float4 d = a - b;
    d.x -= box.x * rintf(d.x / box.x);
    d.y -= box.y * rintf(d.y / box.y);
    d.z -= box.z * rintf(d.z / box.z);
    d.w = 0.0f;
    return d;
};

__host__ __device__ __forceinline__
float4 wrap_float4(float4 p, float3 box_origin, float3 box)
{
    float rx = p.x - box_origin.x;  rx -= box.x * floorf(rx / box.x);  p.x = box_origin.x + rx;
    float ry = p.y - box_origin.y;  ry -= box.y * floorf(ry / box.y);  p.y = box_origin.y + ry;
    float rz = p.z - box_origin.z;  rz -= box.z * floorf(rz / box.z);  p.z = box_origin.z + rz;
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

struct Particle{
    float4 position;
    float4 polarity;
    float4 position_force;
    float4 polarity_force;
};

__global__ void particleSimulation_compute_pcell_ids(
    const Particle* __restrict__ particles,
    int* pcell_ids,
    float3 box_origin,
    float cell_size,
    int3 grid_size,
    int num_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float4 pos = particles[idx].position;
    int3 cell;
    cell.x = (int)floorf((pos.x - box_origin.x) / cell_size);
    cell.y = (int)floorf((pos.y - box_origin.y) / cell_size);
    cell.z = (int)floorf((pos.z - box_origin.z) / cell_size);
    // apply periodic boundary conditions
    cell.x = pmod(cell.x, grid_size.x);
    cell.y = pmod(cell.y, grid_size.y);
    cell.z = pmod(cell.z, grid_size.z);

    int pcell_id = cell.x + cell.y * grid_size.x + cell.z * grid_size.x * grid_size.y;
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
                                float3 box_origin,
                                float3 box,
                                int3 grid_size,
                                float cutoff_radius,
                                float force_strength,
                                float force_r0,
                                float force_alpha,
                                Particle* particles,
                                int num_particles)
{
    int r = blockIdx.x;
    if(r >= num_occupied_cells) return;

    int home_cell_id = cell_ids[r];
    int home_start =  cell_start[home_cell_id];
    if(home_start < 0 || home_start >= num_particles) return;
    int home_end = cell_end[home_cell_id];
    int3 home_cell = make_int3(0,0,0);
    home_cell.x = home_cell_id % grid_size.x;
    home_cell.y = (home_cell_id / grid_size.x) % grid_size.y;
    home_cell.z = home_cell_id / (grid_size.x * grid_size.y);



    float cutoff_radius2 = cutoff_radius * cutoff_radius;
    float exp_term_at_cutoff = expf(-(cutoff_radius - force_r0) * force_alpha);
    //int num_cells = grid_size.x * grid_size.y * grid_size.z;


    for(int i = threadIdx.x + home_start; i < home_end; i += blockDim.x)
    {
       Particle particle_i = particles[i];
       float4 pos_i = particle_i.position;
       float4 pol_i = particle_i.polarity;
       float4 pos_force = make_float4(0.0f);
       float4 pol_force = make_float4(0.0f);

       //loop over neighboring cells
        for(int dz = -1; dz <= 1; dz++)
        for(int dy = -1; dy <= 1; dy++)
        for(int dx = -1; dx <= 1; dx++)
        {
            int3 neighbor_cell;
            neighbor_cell.x = pmod(home_cell.x + dx, grid_size.x);
            neighbor_cell.y = pmod(home_cell.y + dy, grid_size.y);
            neighbor_cell.z = pmod(home_cell.z + dz, grid_size.z);
            int neighbor_cell_id = neighbor_cell.x + neighbor_cell.y * grid_size.x + neighbor_cell.z * grid_size.x * grid_size.y;
            int neighbor_start = cell_start[neighbor_cell_id];
            if(neighbor_start == -1) continue; //empty cell
            int neighbor_end = cell_end[neighbor_cell_id];
            for(int j = neighbor_start; j < neighbor_end; j++)
            {
                if(j == i) continue; //skip self
                Particle particle_j = particles[j];
                float4 pos_j = particle_j.position;
                float4 d = wrapped_float4_diff(pos_i, pos_j, box);
                float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                if(r2 > cutoff_radius2) continue;
                float4 pol_j = particle_j.polarity;
                float polarity_dot = pol_i.x * pol_j.x + pol_i.y * pol_j.y + pol_i.z * pol_j.z;
                float r = sqrtf(r2);
                float exp_term = expf(-(r - force_r0) * force_alpha);
                float force_magnitude =  (exp_term*exp_term - polarity_dot*exp_term);
                //force_magnitude -= (exp_term_at_cutoff*exp_term_at_cutoff - polarity_dot*exp_term_at_cutoff);
                force_magnitude *= 2.0f*force_strength;
                pos_force += d * (force_magnitude/r);
            }
        }
        particles[i].position_force = pos_force;
        particles[i].polarity_force = pol_force;
    }
};

__global__ void particleSimulation_loop_tiled(
                                const int* __restrict__ cell_ids,
                                const int* __restrict__ cell_start,
                                const int* __restrict__ cell_end,
                                int num_occupied_cells,
                                float3 box_origin,
                                float3 box,
                                int3 grid_size,
                                float cutoff_radius,
                                Particle* particles,
                                int num_particles)
{
    int r = blockIdx.x;
    if(r >= num_occupied_cells) return;

    int home_cell_id = cell_ids[r];
    int home_start =  cell_start[home_cell_id];
    if(home_start < 0 || home_start >= num_particles) return;
    int home_end = cell_end[home_cell_id];
    int3 home_cell = make_int3(0,0,0);
    home_cell.x = home_cell_id % grid_size.x;
    home_cell.y = (home_cell_id / grid_size.x) % grid_size.y;
    home_cell.z = home_cell_id / (grid_size.x * grid_size.y);



    float cutoff_radius2 = cutoff_radius * cutoff_radius;
    float force_strength = 1.0f;
    float force_minimum = 0.5f*cutoff_radius;
    float force_scale = 1.0f/cutoff_radius;
    float exp_term_at_cutoff = expf(-(cutoff_radius - force_minimum) * force_scale);
    int num_cells = grid_size.x * grid_size.y * grid_size.z;

    extern __shared__ unsigned char shared_mem_tile[];
    float4* home_positions = reinterpret_cast<float4*>(shared_mem_tile);
    float4* home_polarities = home_positions + TILE_DIM;

    //split particles in tiles of size TILE_DIM
    for(int h0 = home_start; h0 < home_end; h0 += TILE_DIM)
    {
        int num_particles_in_home_tile = min(TILE_DIM, home_end - h0);

        if(threadIdx.x < num_particles_in_home_tile) {
            Particle p = particles[h0 + threadIdx.x];
            home_positions[threadIdx.x] = p.position;
            home_polarities[threadIdx.x] = p.polarity;
        };
        __syncthreads();

        int ti = threadIdx.x;
        if(ti < num_particles_in_home_tile) {
            int i = h0 + ti;
            float4 pos_i = home_positions[ti];
            float4 pol_i = home_polarities[ti];
            float4 pos_force = make_float4(0.0f);
            float4 pol_force = make_float4(0.0f);

            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int3 neighbor_cell;
                neighbor_cell.x = pmod(home_cell.x + dx, grid_size.x);
                neighbor_cell.y = pmod(home_cell.y + dy, grid_size.y);
                neighbor_cell.z = pmod(home_cell.z + dz, grid_size.z);
                int neighbor_cell_id = neighbor_cell.x + neighbor_cell.y * grid_size.x + neighbor_cell.z * grid_size.x * grid_size.y;
                int neighbor_start = cell_start[neighbor_cell_id];
                if (neighbor_start == -1) continue; // empty cell
                int neighbor_end = cell_end[neighbor_cell_id];
                for(int j = neighbor_start; j < neighbor_end; j++)
                {
                    if(j == i) continue; //skip self
                    Particle particle_j = particles[j];
                    float4 pos_j = particle_j.position;
                    float4 pol_j = particle_j.polarity;
                    float polarity_dot = pol_i.x * pol_j.x + pol_i.y * pol_j.y + pol_i.z * pol_j.z;
                    float4 d = wrapped_float4_diff(pos_i, pos_j, box);
                    float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                    if(r2 > cutoff_radius2) continue;
                    
                    float r = sqrtf(r2);
                    float exp_term = expf(-(r - force_minimum) * force_scale);
                    float force_magnitude =  (exp_term*exp_term - 2*polarity_dot*exp_term);
                    force_magnitude -= exp_term_at_cutoff*exp_term_at_cutoff - 2*polarity_dot*exp_term_at_cutoff;
                    force_magnitude *= force_strength;
                    pos_force += d * (force_magnitude/r);
                }
            };

            particles[i].position_force = pos_force;
            particles[i].polarity_force = pol_force;
        };

           
        
    }
};

__global__ void particleSimulation_step(
    Particle* particles,
    int num_particles,
    float3 box_origin, float3 box,
    float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Update positions and polarities based on forces
    particles[idx].position += particles[idx].position_force * dt;
    particles[idx].polarity += particles[idx].polarity_force * dt;
    particles[idx].polarity = normalize(particles[idx].polarity);
    particles[idx].position = wrap_float4(particles[idx].position, box_origin, box);
    // Reset forces for the next iteration
    particles[idx].position_force = make_float4(0.0f);
    particles[idx].polarity_force = make_float4(0.0f);
};

class ParticleSimulation {
public:
    //system parameters
    float3 box_origin;
    float3 box;
    float cell_size;
    int3 grid_size;
    int num_cells;
    float skin;
    float cutoff_radius;
    //cuda parameters
    int num_threads_per_block = 64;
    //simulation variables
    Particle* particles = nullptr;
    Particle* temp_particles = nullptr;
    float4* position_before_update = nullptr;
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
    CubWorkspace sort_workspace;
    CubWorkspace rle_workspace;
    CubWorkspace scan_workspace;
    CubWorkspace max_workspace;
    //potential params
    float force_strength;
    float force_r0;
    float force_alpha;

    ParticleSimulation(int num_particles, float f_strength, float f_r0, float f_alpha, float3 box_origin, float3 box, float cell_size, float cutoff_radius)
    {
        this->force_strength = f_strength;
        this->force_r0 = f_r0;
        this->force_alpha = f_alpha;
        this->box_origin = box_origin;
        this->box = box;
        this->cell_size = cell_size;
        this->cutoff_radius = cutoff_radius;
        this->skin = fmaxf(cell_size - cutoff_radius, 0.0f);
        grid_size.x = (int)ceilf((box.x - box_origin.x) / cell_size);
        grid_size.y = (int)ceilf((box.y - box_origin.y) / cell_size);
        grid_size.z = (int)ceilf((box.z - box_origin.z) / cell_size);
        num_cells = grid_size.x * grid_size.y * grid_size.z;
        this->num_particles = num_particles;
        allocate();
    };

    void allocate()
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

        //cudaMalloc(&position_before_update, num_particles * sizeof(float4));
        //cudaMalloc(&displacements_from_update, num_particles * sizeof(float));
    };

    void random_init(default_random_engine& rng, float3 min_pos, float3 max_pos)
    {
        uniform_real_distribution<float> dist_x(max(min_pos.x, box_origin.x), min(max_pos.x, box_origin.x + box.x));
        uniform_real_distribution<float> dist_y(max(min_pos.y, box_origin.y), min(max_pos.y, box_origin.y + box.y));
        uniform_real_distribution<float> dist_z(max(min_pos.z, box_origin.z), min(max_pos.z, box_origin.z + box.z));
        normal_distribution <float> dist_p(0.0f, 1.0f);

        Particle* h_particles = new Particle[num_particles];
        for(int i = 0; i < num_particles; i++)
        {
            h_particles[i].position = make_float4(dist_x(rng), dist_y(rng), dist_z(rng), 0.0f);
            float3 p = make_float3(dist_p(rng), dist_p(rng), dist_p(rng));
            p = normalize(p);
            h_particles[i].polarity = make_float4(p.x, p.y, p.z, 0.0f);
            h_particles[i].position_force = make_float4(0.0f);
            h_particles[i].polarity_force = make_float4(0.0f);
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
            force_strength,
            force_r0,
            force_alpha,
            particles,
            num_particles);
    };

    void loop_tiled()
    {
        size_t shmem = 2 * TILE_DIM * sizeof(float4);
        particleSimulation_loop_tiled<<<num_occupied_cells, TILE_DIM, shmem>>>(
            cell_ids,
            cell_start,
            cell_end,
            num_occupied_cells,
            box_origin,
            box,
            grid_size,
            cutoff_radius,
            particles,
            num_particles);
    }
    
    void step(float dt)
    {
        int num_blocks = (num_particles + num_threads_per_block - 1) / num_threads_per_block;
        particleSimulation_step<<<num_blocks, num_threads_per_block>>>(
            particles,
            num_particles,
            box_origin, box,
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