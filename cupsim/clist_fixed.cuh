
__global__ void build_cell_list_fixed(__restrict__ int* cell_ids,
                                     int* cell_counts,
                                     int* cell_list,
                                     int* cell_overflow_flags,
                                     int num_particles,
                                     int num_cells,
                                     int max_particles_per_cell)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        const int cid = cell_ids[idx];
        //check if cell id is valid
        if (cid >= 0 && cid < num_cells) 
        {
            //increase the number of particles in the cell atomically
            int slot = atomicAdd(&cell_counts[cid], 1);
            //check if overflow occurred (i.e. we have more particles than max_particles_per_cell)
            //the loop will continue as normal but we will not write the particle index to the cell list
            //the overflow flag will be used to rebuild the cell list with a larger max_particles_per_cell
            if(slot < max_particles_per_cell) 
            {
                cell_list[cid * max_particles_per_cell + slot] = idx;
            } 
            else 
            {
                atomicExch(&cell_overflow_flags[cid], 1); // mark overflow
            }
        }
    }
};

class CellListFixed
{

    int num_particles = 0;
    int max_particles_per_cell = 0;
    System system;
    int* cell_ids = nullptr;
    int* cell_counts = nullptr;
    int* cell_list = nullptr;
    int* particle_cells = nullptr;
    int num_cells = 0;
    int num_blocks = 0;
    int num_threads = 0;



public: 
    CellListFixed(int num_particles, System system) : num_particles(num_particles), system(system)
    {
        cudaMalloc(&cell_ids, num_particles * sizeof(int));
        float3 box = system.domain_max - system.domain_min;
        float volume = box.x * box.y * box.z;
        float initial_density = system.num_particles*(4.0f/3.0f)*M_PI*system.cutoff*system.cutoff*system.cutoff / volume;
        max_particles_per_cell = (int)(initial_density * 2.0f); // safety factor of 2
        num_cells = system.grid_size.x * system.grid_size.y * system.grid_size.z;
    };

    void update()
    {
        cudaStream_t stream; 
        cudaStreamCreate(&stream);
        cudaMemsetAsync(cell_counts, 0, num_cells * sizeof(unsigned int), stream);
        cudaMemsetAsync(cell_ids, 0, num_particles * sizeof(int), stream);
        
    };
}