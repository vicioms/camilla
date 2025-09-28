#include <random>
#include <iostream>
#include <fstream>
#include "simulation.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main() {
    // Initialize GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Particle Simulation Animation", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glPointSize(2.0f);

    // CUDA device info
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::printf("%s %d.%d\n", deviceProp.name, deviceProp.major, deviceProp.minor);

    // Setup simulation
    System system;
    create_system(make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f), 1.0f, 1.1f, true, system);
    
    default_random_engine generator(42);
    int num_particles = 10000;
    Simulation sim(system, num_particles);
    sim.particles_random_init(generator);
    
    printf("Starting animated simulation with %d particles\n", num_particles);
    printf("Controls: ESC to exit\n");

    float dt = 0.01f;
    int step = 0;
    auto time_start = std::chrono::high_resolution_clock::now();
    
    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // Clear screen
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Setup view matrix (simple camera)
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = 1200.0f / 900.0f;
        glOrtho(-6.0f * aspect, 6.0f * aspect, -6.0f, 6.0f, -20.0f, 20.0f);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        // Simple rotation for better viewing
        float time = glfwGetTime();
        glRotatef(20.0f, 1.0f, 0.0f, 0.0f);
        glRotatef(time * 10.0f, 0.0f, 0.0f, 1.0f);
        glTranslatef(-5.0f, -5.0f, -5.0f);

        // Update simulation
        sim.update();
        sim.loop();
        sim.step(dt);

        // Get particles for rendering
        Particle* h_particles;
        sim.get_particles(h_particles);

        // Render particles
        glBegin(GL_POINTS);
        for(int i = 0; i < num_particles; i++) {
            float3 pos = h_particles[i].position;
            float3 pol = h_particles[i].polarity;
            
            // Color based on polarity direction
            float r = (pol.x + 1.0f) * 0.5f;
            float g = (pol.y + 1.0f) * 0.5f;
            float b = (pol.z + 1.0f) * 0.5f;
            glColor3f(r, g, b);
            
            glVertex3f(pos.x, pos.y, pos.z);
        }
        glEnd();

        delete[] h_particles;

        // Print progress occasionally
        if(step % 1000 == 0) {
            auto time_now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float, std::milli>(time_now - time_start).count();
            printf("Step %d, Elapsed: %.2f s\r", step, elapsed/1000);
            fflush(stdout);
        }

        step++;

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    sim.free();
    glfwTerminate();
    return 0;
}