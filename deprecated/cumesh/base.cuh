#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

float* copy_to_device(const float* h_data, size_t count)
{
    float* d_data;
    cudaMalloc(&d_data, sizeof(float) * count);
    cudaMemcpy(d_data, h_data, sizeof(float) * count, cudaMemcpyHostToDevice);
    return d_data;
};
float* copy_to_host(const float* d_data, size_t count)
{
    float* h_data = (float*)malloc(sizeof(float) * count);
    cudaMemcpy(h_data, d_data, sizeof(float) * count, cudaMemcpyDeviceToHost);
    return h_data;
};
__global__ void zero(float* v, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
    {
        return;
    };
    v[idx] = 0.0f;
};
__global__ void zero(float3* v, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
    {
        return;
    };
    v[idx].x = 0.0f;
    v[idx].y = 0.0f;
    v[idx].z = 0.0f;
};
__host__ __device__ int pmod(int i, int n)
{
    return (i % n + n) % n;
}
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
};
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
};
__host__ __device__ inline float3 operator*(const float3& a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
};
__host__ __device__ inline float3 operator*(const float s, const float3& a) {
    return a * s;
};
__host__ __device__ inline float3 operator/(const float3& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
};
__device__ __host__ inline float3 cross(const float3& u, const float3& v)
{
    return make_float3(u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x);
};
__device__ __host__ inline float dot(const float3& u, const float3& v)
{
    return u.x*v.x + u.y*v.y + u.z*u.z;
};
__device__ __host__ inline float3 rodrigues(const float3& v, const float3& n, const float theta)
{
    float3 vec1 = cross(n,v);
    float3 vec2 = cross(n, vec1);
    float coef1 = sin(theta);
    float coef2 = 1- cos(theta);
    return v + coef1*vec1 + coef2*vec2;
};
__host__ __device__ inline float length(const float3& a) {
    return sqrtf(dot(a, a));
};
__host__ __device__ inline float3 normalize(const float3& a) {
    return a / length(a);
};

__host__ __device__ inline float cotangent(float3 u, float3 v) {
    float dot_uv = dot(u, v);
    float3 cross_uv = cross(u, v);
    float cross_norm = length(cross_uv);
    return dot_uv / cross_norm;  // avoid divide-by-zero
}




