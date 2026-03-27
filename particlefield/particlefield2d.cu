#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

int main() {
    const int NX = 256;
    const int NY = 256;
    
    // Allocate device memory
    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, NX * NY * sizeof(cufftComplex));
    
    // Initialize data on host and copy to device
    cufftComplex *h_data = new cufftComplex[NX * NY];
    for (int i = 0; i < NX * NY; i++) {
        h_data[i].x = 1.0f;
        h_data[i].y = 0.0f;
    }
    cudaMemcpy(d_data, h_data, NX * NY * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    // Create FFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, NX, NY, CUFFT_C2C);
    
    // Example 1: Forward FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    
    std::cout << "Forward FFT completed" << std::endl;
    
    // Example 2: Inverse FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    
    std::cout << "Inverse FFT completed" << std::endl;
    
    // Copy result back to host
    cudaMemcpy(h_data, d_data, NX * NY * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    delete[] h_data;
    
    return 0;
}