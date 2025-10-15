#include <iostream>
#include <random>
#include "psimulation2d.cuh"
#include <chrono>
#include <fstream>
using namespace std;

int main()
{
    int num_particles = 1 << 9; 
    float2 box_origin = make_float2(0.0f, 0.0f);
    float2 box_size = make_float2(12.0f, 12.0f);
    float cutoff_radius = 1.0f;
    float skin = 0.2*cutoff_radius;
    float cell_size = cutoff_radius + skin;
    int num_steps = 100000;
    float dt = 0.001f;
    float force_strength = 0.5f;
    float v0 = 0.1f;
    float eta = 0.1f;
    default_random_engine eng(42);
    ParticleSimulation sim(num_particles, v0, force_strength, box_origin, box_size, cell_size, cutoff_radius, eta, 42);
    sim.random_init(eng, box_origin, box_origin + box_size);
    sim.set_threads_per_block(64);
    float total_time = 0;
    auto time_start = std::chrono::high_resolution_clock::now();
    for(int step = 0; step < num_steps; step++) {
        sim.updateCellList();
        sim.loop();
        sim.step(dt);
        if(step % 10000 == 0) {
            cout << "Step " << step << "\n";
        }
    };
    auto time_end = std::chrono::high_resolution_clock::now();
    total_time = std::chrono::duration<float, std::milli>(time_end - time_start).count();
    cout << "Average time per step: " << total_time/num_steps << " ms\n";

    
    Particle* h_particles = new Particle[num_particles];
    cudaMemcpy(h_particles, sim.particles, sizeof(Particle) * num_particles, cudaMemcpyDeviceToHost);
    ofstream outfile("particles2d.txt");
    for(int i = 0; i < num_particles; i++) {
        outfile << h_particles[i].position.x << " " << h_particles[i].position.y << " " << h_particles[i].theta << " " << h_particles[i].shape.xx << " " << h_particles[i].shape.xy << " " << h_particles[i].shape.yy << "\n";
    };
    outfile.close();
    delete[] h_particles;
    sim.free();
    return 0;
};