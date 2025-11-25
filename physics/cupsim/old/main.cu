#include <random>
#include <iostream>
#include <fstream>
#include "clist.cuh"
#include <chrono>
#include <cuda_runtime.h>
using namespace std;
int main() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::printf("%s %d.%d\n", deviceProp.name, deviceProp.major, deviceProp.minor);
    System system;
    create_system(make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f), 1.0f, 1.0f, true, system);
    default_random_engine generator(42);
    int num_particles = 10000;
    Simulation sim(system, num_particles);
    sim.particles_random_init(generator);
    printf("Starting simulation with %d particles\n", num_particles);
    auto time_start = std::chrono::high_resolution_clock::now();
    int num_steps = 1000000;
    float dt = 0.01f;
    for(int step = 0; step < num_steps; step++) {
        if(step % 10000 == 0) {
            printf("Step %d / %d\n", step, num_steps);
            auto time_now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float, std::milli>(time_now - time_start).count();
            printf("Elapsed time: %f s\n", (elapsed/1000));
        };
        sim.update();
        sim.loop();
        sim.step(dt);
    }//
    auto time_end = std::chrono::high_resolution_clock::now();
    float total_runtime = std::chrono::duration<float, std::milli>(time_end - time_start).count();
    printf("Total runtime: %f s\n", (total_runtime/1000));
    Particle* h_particles;
    sim.get_particles(h_particles);
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