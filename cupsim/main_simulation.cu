#include <iostream>
#include <random>
#include "psimulation.cuh"
#include <chrono>
#include <fstream>
using namespace std;

int main()
{
    int num_particles = 1024;
    float3 box_origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 box_size = make_float3(4.8f, 4.8f, 4.8f);
    float cutoff_radius = 1.0f;
    float skin = 0.2*cutoff_radius;
    float cell_size = cutoff_radius + skin;
    int num_steps = 1000000;
    float dt = 0.01f;
    float force_strength = 1.0f;
    float force_r0 = 0.5f*cutoff_radius;
    float force_alpha = 5.0f/cutoff_radius;
    default_random_engine eng(42);
    ParticleSimulation sim(num_particles, force_strength, force_r0, force_alpha, box_origin, box_size, cell_size, cutoff_radius);
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
    ofstream outfile("particles.txt");
    for(int i = 0; i < num_particles; i++) {
        outfile << h_particles[i].position.x << " " << h_particles[i].position.y << " " << h_particles[i].position.z;
        outfile << " " << h_particles[i].polarity.x << " " << h_particles[i].polarity.y << " " << h_particles[i].polarity.z << "\n";
    };
    outfile.close();
    delete[] h_particles;
    sim.free();
    return 0;
};