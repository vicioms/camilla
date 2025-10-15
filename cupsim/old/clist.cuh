#pragma once
#include "basics.cuh"
#include "cuda_kernels.cuh"
#include <cub/cub.cuh>

__global__ void scatter_ranges(const int* __restrict__ unique_ids,
                               const int* __restrict__ unique_offsets,   // start per run
                               const int* __restrict__ unique_counts,    // count per run
                               int num_unique_ids,
                               int* __restrict__ start,      // dense
                               int* __restrict__ end)        // dense (exclusive)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_unique_ids) {
        int cid = unique_ids[r];
        int s   = unique_offsets[r];
        int e   = s + unique_counts[r];
        start[cid] = s;
        end[cid]   = e;
    }
};

__global__ void neighbor_loop_kernel(const int* __restrict__ occupied_cells_ids,
                                     const int* __restrict__ offsets,
                                     const int* __restrict__ counts,
                                     const int num_occupied_cells,
                                     const int* __restrict__ cell_start,
                                     const int* __restrict__ cell_end,
                                     Particle* __restrict__ particles,
                                     int num_particles,
                                     System system)
{
    //int r = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.x;
    if(r >= num_occupied_cells) return;
    //int cell_id = occupied_cells_ids[r];
    int start = offsets[r];
    int end = start + counts[r];

    float cutoff2= system.cutoff * system.cutoff;
    float cutoff = system.cutoff;
    float3 domain_min = system.domain_min;
    float3 domain_max = system.domain_max;
    //for(int i = start; i < end; i++)
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
    {
        int nb_count = 0;
        float3 particle_position = particles[i].position;
        float3 particle_polarity = particles[i].polarity;
        int3 particle_cell = compute_cell(particle_position, system);
        float3 vel = make_float3(0.0f, 0.0f, 0.0f);
        for(int dz = -1; dz <= 1; dz++) {
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    int3 neighbor_cell;
                    if(system.use_periodic) {
                        neighbor_cell = make_int3_periodic(particle_cell.x + dx, particle_cell.y + dy, particle_cell.z + dz, system.grid_size);
                    }
                    else
                    {
                        neighbor_cell = make_int3(particle_cell.x + dx, particle_cell.y + dy, particle_cell.z + dz);
                        if(neighbor_cell.x < 0 || neighbor_cell.x >= system.grid_size.x) continue;
                        if(neighbor_cell.y < 0 || neighbor_cell.y >= system.grid_size.y) continue;
                        if(neighbor_cell.z < 0 || neighbor_cell.z >= system.grid_size.z) continue;
                    }
                    int neighbor_cell_id = compute_cell_id(neighbor_cell, system);
                    int n_start = cell_start[neighbor_cell_id];
                    int n_end = cell_end[neighbor_cell_id];
                    if(n_start == -1 || n_end == -1) continue;
                    for(int j = n_start; j < n_end; j++) {
                        if(i == j) continue;
                        float3 r;
                        float3 neighbor_position = particles[j].position;
                        if(system.use_periodic) {
                            r = periodic_diff(particle_position, neighbor_position, domain_min, domain_max);
                        }
                        else {
                            r = particle_position - neighbor_position;
                        };
                        float r_dist2 = dot(r, r);
                        if(r_dist2 < cutoff2) {
                            nb_count++;
                            float3 neighbor_polarity = particles[j].polarity;
                            float s = dot(particle_polarity, neighbor_polarity);
                            morse_force(r,1.0,cutoff/2.0, cutoff/2.0, cutoff, s, vel, true);
                        };
                    };
                }
            }
        }
        particles[i].velocity += vel;
    };
};

class CellList
{
public:
    bool initialized = false;
    //device variables
    int num_particles = 0;
    int num_cells = 0;
    int block_size = 0;
    int grid_size = 0;
    int* d_cell_ids = nullptr;
    int* d_sorted_cell_ids = nullptr;
    int* d_unique_cell_ids = nullptr;
    int* d_unique_cell_offsets = nullptr;
    int* d_unique_cell_counts = nullptr;
    int h_num_unique_cells = 0;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;
    //temporary storages for cub sorting
    CubWorkspace sort_workspace;
    CubWorkspace rle_workspace;
    CubWorkspace scan_workspace;



    CellList() = default;
    CellList(int num_particles, int num_cells, int block_size = 256) : num_particles(num_particles), block_size(block_size) {
        init(num_particles, num_cells);
        grid_size = (num_particles + block_size - 1) / block_size;
    };

    void init(int num_particles, int num_cells, int block_size = 256)
    {
        if(initialized) {
            free();
        }
        this->num_particles = num_particles;
        this->num_cells = num_cells;
        this->block_size = block_size;
        this->grid_size = (num_particles + block_size - 1) / block_size;
        this->h_num_unique_cells = 0;
        device_allocate(d_cell_ids, num_particles);
        device_allocate(d_sorted_cell_ids, num_particles);
        device_allocate(d_unique_cell_ids, num_particles);
        device_allocate(d_unique_cell_counts, num_particles);
        device_allocate(d_unique_cell_offsets, num_particles);
        device_allocate(d_cell_start, num_cells);
        device_allocate(d_cell_end, num_cells);
        initialized = true;
    };
    
    bool update(Particle* particles, Particle* temp_particles, System system)
    {
        if(!initialized) {
            return false;
        };

        compute_cell_ids<<<grid_size, block_size>>>(particles, d_cell_ids, num_particles, system);
        cudaDeviceSynchronize();

        //sorting part
        size_t sort_required_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr,
            sort_required_bytes, 
            d_cell_ids, 
            d_sorted_cell_ids, 
            particles, 
            temp_particles, 
            num_particles);
        sort_workspace.ensure_size(sort_required_bytes);
        cub::DeviceRadixSort::SortPairs(sort_workspace.temp_storage,
            sort_workspace.temp_storage_bytes,
            d_cell_ids,
            d_sorted_cell_ids,
            particles, 
            temp_particles,
            num_particles);
        //RLE part
        size_t rle_required_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr, 
            rle_required_bytes,
            d_sorted_cell_ids,
            d_unique_cell_ids,
            d_unique_cell_counts,
            d_unique_cell_offsets,
            num_particles);

        rle_workspace.ensure_size(rle_required_bytes);
        int* d_num_runs_out = nullptr;
        cudaMalloc(&d_num_runs_out, sizeof(int));
        cub::DeviceRunLengthEncode::Encode(rle_workspace.temp_storage,
            rle_workspace.temp_storage_bytes,
            d_sorted_cell_ids,
            d_unique_cell_ids,
            d_unique_cell_counts,
            d_num_runs_out,
            num_particles);
        h_num_unique_cells = 0;
        cudaMemcpy(&h_num_unique_cells, d_num_runs_out, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_num_runs_out);
        //scan part
        size_t scan_required_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr,
            scan_required_bytes,
            d_unique_cell_counts,
            d_unique_cell_offsets,
            h_num_unique_cells);
        scan_workspace.ensure_size(scan_required_bytes);
        cub::DeviceScan::ExclusiveSum(scan_workspace.temp_storage,
            scan_workspace.temp_storage_bytes,
            d_unique_cell_counts,
            d_unique_cell_offsets,
            h_num_unique_cells);

        cudaMemset(d_cell_start, 0xFF, num_cells * sizeof(int));
        cudaMemset(d_cell_end, 0xFF, num_cells * sizeof(int));

        int runs_grid = (h_num_unique_cells + block_size - 1) / block_size;
        scatter_ranges<<<runs_grid, block_size>>>(d_unique_cell_ids,
            d_unique_cell_offsets, 
            d_unique_cell_counts,
            h_num_unique_cells, 
            d_cell_start, 
            d_cell_end);
        return true;
    };

    void loop(Particle* particles, System system)
    {
        if(!initialized) {
            return;
        };
        //int occupied_grid = (h_num_unique_cells + block_size - 1) / block_size;
        //neighbor_loop_kernel<<<occupied_grid, block_size>>>
        neighbor_loop_kernel<<<h_num_unique_cells, block_size>>>
        (d_unique_cell_ids,
            d_unique_cell_offsets,
            d_unique_cell_counts,
            h_num_unique_cells,
            d_cell_start,
            d_cell_end,
            particles,
            num_particles,
            system);
        //int* h_neighbor_counts = new int[num_particles];
        //cudaMemcpy(h_neighbor_counts, d_neighbor_counts, num_particles * sizeof(int), cudaMemcpyDeviceToHost);
        //for(int i = 0; i < num_particles; i++) {
        //    printf("Particle %d has %d neighbors\n", i, h_neighbor_counts[i]);
        //}
        //delete[] h_neighbor_counts;
    };

   


    void free()
    {
        if(!initialized) {
            return;
        }
        num_particles = 0;
        num_cells = 0;
        block_size = 0;
        grid_size = 0;
        h_num_unique_cells = 0;
        cudaFree(d_cell_ids);
        cudaFree(d_sorted_cell_ids);
        cudaFree(d_unique_cell_ids);
        cudaFree(d_unique_cell_counts);
        cudaFree(d_unique_cell_offsets);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        sort_workspace.free();
        rle_workspace.free();
        scan_workspace.free();
        initialized = false;
    };

    void print_state()
    {
        if(!initialized) {
            printf("CellList not initialized\n");
            return;
        }
        printf("CellList state:\n");
        printf("  num_particles: %d\n", num_particles);
        printf("  block_size: %d\n", block_size);
        printf("  grid_size: %d\n", grid_size);
        printf("  initialized: %s\n", initialized ? "true" : "false");

           
        int* h_unique_cell_ids = new int[h_num_unique_cells];
        int* h_unique_cell_counts = new int[h_num_unique_cells];
        int* h_unique_cell_offsets = new int[h_num_unique_cells];
        
        
        cudaMemcpy(h_unique_cell_ids, d_unique_cell_ids, h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_unique_cell_counts, d_unique_cell_counts, h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_unique_cell_offsets, d_unique_cell_offsets, h_num_unique_cells * sizeof(int), cudaMemcpyDeviceToHost);
        int* h_cell_start = new int[num_cells];
        int* h_cell_end = new int[num_cells];
        cudaMemcpy(h_cell_start, d_cell_start, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cell_end, d_cell_end, num_cells * sizeof(int), cudaMemcpyDeviceToHost);      
        printf("  Cell start: ");
        for(int i = 0; i < num_cells; i++) {
            printf("%d ", h_cell_start[i]);
        };
        printf("\n");
        printf("  Cell end: ");
        for(int i = 0; i < num_cells; i++) {
            printf("%d ", h_cell_end[i]);
        };
        printf("\n");
        delete[] h_cell_start;
        delete[] h_cell_end;
        delete[] h_unique_cell_ids;
        delete[] h_unique_cell_counts;
        delete[] h_unique_cell_offsets;
        
    }
};