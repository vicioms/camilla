#include "../hmath.h"
#include <random>
#include <chrono>
using namespace std;

struct Particle
{
    float3 position;
    float3 polarity;
    float3 position_force;
    float3 polarity_force;
};

class ParticleSimulation
{
public:
    //system parameters
    float3 box_origin;
    float3 box;
    float cell_size;
    int3 grid_size;
    int num_cells;
    float cutoff_radius;
    //simulation variables
    Particle* particles = nullptr;
    int num_particles = 0;
    //cell list variables
    int* cell_heads = nullptr;
    int* next_particle = nullptr;
    
    void allocate()
    {
        particles = (Particle*)malloc(num_particles * sizeof(Particle));
        cell_heads = (int*)malloc(num_cells * sizeof(int));
        next_particle = (int*)malloc(num_particles * sizeof(int));
    };

    void random_init(default_random_engine& rng, float3 min_pos, float3 max_pos)
    {
        uniform_real_distribution<float> dist_x(max(min_pos.x, box_origin.x), min(max_pos.x, box_origin.x + box.x));
        uniform_real_distribution<float> dist_y(max(min_pos.y, box_origin.y), min(max_pos.y, box_origin.y + box.y));
        uniform_real_distribution<float> dist_z(max(min_pos.z, box_origin.z), min(max_pos.z, box_origin.z + box.z));
        normal_distribution <float> dist_p(0.0f, 1.0f);

        for(int i = 0; i < num_particles; i++)
        {
            particles[i].position = make_float3(dist_x(rng), dist_y(rng), dist_z(rng));
            float3 p = make_float3(dist_p(rng), dist_p(rng), dist_p(rng));
            p = normalize(p);
            particles[i].polarity = p;
            particles[i].position_force = make_float3(0.0f, 0.0f, 0.0f);
            particles[i].polarity_force = make_float3(0.0f, 0.0f, 0.0f);
        }
    };

    void updateCellList()
    {
        for(int c = 0; c < num_cells; c++) cell_heads[c] = -1;
        for(int i =0; i < num_particles; i++)
        {
            int3 cell;
            cell.x = (int)floorf((particles[i].position.x - box_origin.x) / cell_size);
            cell.y = (int)floorf((particles[i].position.y - box_origin.y) / cell_size);
            cell.z = (int)floorf((particles[i].position.z - box_origin.z) / cell_size);
            int cell_id = cell.x + cell.y * grid_size.x + cell.z * grid_size.x * grid_size.y;
            next_particle[i] = cell_heads[cell_id];
            cell_heads[cell_id] = i;
        }
    };

    void loop()
    {
        for(int c = 0; c < num_cells; c++)
        {
            if(cell_heads[c] == -1) continue;
            int3 cell;
            cell.x = c % grid_size.x;
            cell.y = (c / grid_size.x) % grid_size.y;
            cell.z = c / (grid_size.x * grid_size.y);
            int i = cell_heads[c];
            while(i != -1)
            {
                for(int dx = -1; dx <= 1; dx++)
                {
                    for(int dy = -1; dy <= 1; dy++)
                    {
                        for(int dz = -1; dz <= 1; dz++)
                        {
                            int3 neighbor_cell;
                            neighbor_cell.x = pmod(cell.x + dx, grid_size.x);
                            neighbor_cell.y = pmod(cell.y + dy, grid_size.y);
                            neighbor_cell.z = pmod(cell.z + dz, grid_size.z);
                            int neighbor_cell_id = neighbor_cell.x + neighbor_cell.y * grid_size.x + neighbor_cell.z * grid_size.x * grid_size.y;
                            int j = cell_heads[neighbor_cell_id];
                            if(j == -1) continue;
                            while(j != -1)
                            {
                                if(i == j)
                                {
                                    j = next_particle[j];
                                    continue;
                                };
                                //printf("pair: %d %d %f\n", i, j, length(particles[i].position - particles[j].position));
                                j = next_particle[j];
                            }

                        };
                    };
                };
                i = next_particle[i];
            };
        }   
    };

};

int main()
{
    ParticleSimulation sim;
    sim.box_origin = make_float3(0.0f, 0.0f, 0.0f);
    sim.box = make_float3(10.0f, 10.0f, 10.0f);
    sim.cell_size = 1.0f;
    sim.grid_size.x = (int)ceilf(sim.box.x / sim.cell_size);
    sim.grid_size.y = (int)ceilf(sim.box.y / sim.cell_size);
    sim.grid_size.z = (int)ceilf(sim.box.z / sim.cell_size);
    sim.num_cells = sim.grid_size.x * sim.grid_size.y * sim.grid_size.z;
    sim.cutoff_radius = 1.0f;
    sim.num_particles = 20000;
    sim.allocate();
    default_random_engine rng;
    sim.random_init(rng, sim.box_origin, sim.box + sim.box_origin);
    auto start = chrono::high_resolution_clock::now();
    for(int step = 0; step < 10000; step++)
    {
        sim.updateCellList();
        sim.loop();
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float> duration = end - start;
    printf("Time: %f s\n", duration.count());
    
    
    return 0;
}