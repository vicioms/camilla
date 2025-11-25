#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
using namespace std;
const bool only_positive_friction = true;

__device__ __host__ inline int pack_int2(int x, int y, int N)
{
    return y*N + x;
};

__device__ __host__ inline int pack_int2(int2 loc, int N)
{
    return loc.y*N + loc.x;
};

__device__ __host__ inline int pmod(int i, int n)
{
    return (i%n+n)%n;
}

__device__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void add(float* u, float val,  int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x,y,Nx);
    u[i] += val;
};

__global__ void zero(float* u,  int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x,y,Nx);
    u[i] = 0.0f;
};

__global__ void zero(bool* u,  int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x,y,Nx);
    u[i] = false;
};

__global__ void init_random(curandState* states, int Nx, int Ny, unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x,y,Nx);
    unsigned long long thread_seed = seed ^ ((i + 1ULL) * 0x9E3779B97F4A7C15ULL);
    curand_init(thread_seed, 0, 0, &states[i]);
};

__global__ void init_friction(float* f_friction, curandState* rand_states, float fric_m, float fric_s, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x,y,Nx);
    f_friction[i] = curand_normal(&rand_states[i])*fric_s + fric_m;
    if(only_positive_friction)
        f_friction[i] = abs(f_friction[i]);
};

__global__ void find_min_delta(float* f_el, float* f_drive, float* f_friction, float* min_delta, int* min_index, int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x, y, Nx);
    float delta = f_friction[i] - (f_el[i] + f_drive[i]);

    if (delta >= 0.0f) {
        float old = atomicMinFloat(min_delta, delta);
        if (delta < old) {
            atomicExch(min_index, i);
        }
    }
};

__global__ void get_unstable_sites(float* f_el, float* f_drive, float* f_friction, bool* unstable, int* unstable_count,  int Nx, int Ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x, y, Nx);
    
    if(f_friction[i] <= f_el[i] + f_drive[i])
    {
        atomicAdd(unstable_count, 1);
        unstable[i] = true;
    }
    else
    {
        unstable[i] = false;
    }
    return;
};

__global__ void propagate(float* f_el, float* f_drive, float* f_friction, 
    curandState* rand_states,  bool* unstable, 
    float dh, float k, float k0, float fric_m, float fric_s,
     bool periodic_x, bool periodic_y, int Nx, int Ny )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int i = pack_int2(x, y, Nx);
    if(unstable[i] == false)
        return;
    f_friction[i] = curand_normal(&rand_states[i])*fric_s + fric_m;
    if(only_positive_friction)
        f_friction[i] = abs(f_friction[i]);
    f_drive[i] -= k0*dh;
    f_el[i] -= 4*k*dh;
    if(periodic_x)
    {
        atomicAdd(&f_el[pack_int2(pmod(x-1, Nx), y, Nx)], k*dh);
        atomicAdd(&f_el[pack_int2(pmod(x+1, Nx), y, Nx)], k*dh);
    }
    else
    {
        if (x > 0) 
        {
           atomicAdd(&f_el[pack_int2(x-1, y, Nx)], k*dh);
        };
        if (x < Nx -1 ) 
        {
           atomicAdd(&f_el[pack_int2(x+1, y, Nx)], k*dh);
        };
    };

    if(periodic_y)
    {
        atomicAdd(&f_el[pack_int2(x, pmod(y-1, Ny), Nx)], k*dh);
        atomicAdd(&f_el[pack_int2(x, pmod(y+1, Ny), Nx)], k*dh);
    }
    else
    {
        if (y > 0) 
        {
           atomicAdd(&f_el[pack_int2(x, y-1, Nx)], k*dh);
        };
        if (y < Ny -1 ) 
        {
           atomicAdd(&f_el[pack_int2(x, y+1, Nx)], k*dh);
        };
    };
};

int main()
{
    int Nx = 1024;
    int Ny = 1024;
    float* f_el ;
    float* f_drive;
    float* f_friction;
    bool* unstable;
    
    int* min_index;
    float* min_delta;
    curandState* rand_states;
    cudaMalloc(&f_el,sizeof(float)*Nx*Ny);
    cudaMalloc(&f_drive,sizeof(float)*Nx*Ny);
    cudaMalloc(&f_friction,sizeof(float)*Nx*Ny);
    cudaMalloc(&unstable,sizeof(bool)*Nx*Ny);
    cudaMalloc(&min_index, sizeof(int));
    cudaMalloc(&min_delta, sizeof(float));
    cudaMalloc(&rand_states, sizeof(curandState)*Nx*Ny);

    float fric_m = 0.0;
    float fric_s = 1.0;
    float dh = 0.1;
    float k = 1.0;
    float k0 = 0.1;
    bool true_val = true;
    int zero_val = 0;

    int linear_block_dim = 16;
    dim3 blockDim(linear_block_dim, linear_block_dim);
    dim3 gridDim((Nx + linear_block_dim-1) / linear_block_dim, (Ny + linear_block_dim-1) / linear_block_dim);

    init_random<<<gridDim, blockDim>>>(rand_states, Nx, Nx, 1101252);
    init_friction<<<gridDim, blockDim>>>(f_friction, rand_states, fric_m, fric_s, Nx, Ny);
    zero<<<gridDim, blockDim>>>(f_el, Nx,Ny);
    zero<<<gridDim, blockDim>>>(f_drive, Nx,Ny);
    cout << "initialized" << endl;
    int num_avalanches = 100000;
    vector<int> avalanche_sizes(num_avalanches);
    for(int step = 0; step < num_avalanches; step++)
    {
        if(step % 1000 == 0)
        {
            cout << step << endl;
        }
        find_min_delta<<<gridDim, blockDim>>>(f_el, f_drive, f_friction,  min_delta, min_index, Nx, Ny);
        cudaDeviceSynchronize();
        //cout << "found epicenter" << endl;
        float min_delta_host;
        int min_index_host;
        cudaMemcpy(&min_delta_host, min_delta, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&min_index_host, min_index, sizeof(int), cudaMemcpyDeviceToHost);
        add<<<gridDim, blockDim>>>(f_drive, min_delta_host, Nx, Ny);
        
        cudaMemcpy(unstable + min_index_host, &true_val, sizeof(bool), cudaMemcpyHostToDevice);
        //cout << "avalanche starts" << endl;
        int avalanche_size = 1;
        while(true)
        {
            
    
            propagate<<<gridDim, blockDim>>>(f_el, f_drive, f_friction, rand_states, unstable,dh, k,k0, fric_m, fric_s, true, true, Nx, Ny);
            cudaDeviceSynchronize();

            int* unstable_count;
            cudaMalloc(&unstable_count, sizeof(int));
            cudaMemcpy(&unstable_count, &zero_val, sizeof(int), cudaMemcpyDeviceToHost);
            get_unstable_sites<<<gridDim, blockDim>>>(f_el, f_drive, f_friction, unstable, unstable_count, Nx, Ny);
            cudaDeviceSynchronize();
            int count;
            cudaMemcpy(&count, unstable_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (count == 0) break;
            avalanche_size += count;
        };
        avalanche_sizes.push_back(avalanche_size);
        //cout << avalanche_size << endl;
        //cout << "-----" << endl;
        //zero<<<gridDim, blockDim>>>(unstable, Nx, Ny);
        //cudaDeviceSynchronize();
    };
    for(auto s : avalanche_sizes)
    {
        cout << s << endl;
    };
};