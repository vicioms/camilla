#pragma once
#include "base.cuh"
struct trimesh
{
    float3* vertices;
    int3* triangles;
    int* edge_from;
    int* edge_to;
    int* edge_other;
    int* edge_triangle;
    int num_vertices;
    int num_triangles;
    int num_edges;
};
struct polyemb
{
    trimesh* mesh;
    int* start_indices;
    int num_polygons;
    float3* centers;
    float3* area_vectors;
};
__global__ void compute_triangle_area_vectors(const float3* __restrict__ vertices,
                                       const int3* __restrict__ triangles,
                                       float3* __restrict__ area_vectors,
                                       int num_triangles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triangles) return;

    int3 tri = triangles[tid];
    float3 a = vertices[tri.x];
    float3 b = vertices[tri.y];
    float3 c = vertices[tri.z];

    

    area_vectors[tid] = 0.5f * cross(b - a, c - a);
};
__global__ void compute_triangle_areas(const float3* __restrict__ vertices,
                                       const int3* __restrict__ triangles,
                                       float* __restrict__ areas,
                                       int num_triangles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triangles) return;

    int3 tri = triangles[tid];
    float3 a = vertices[tri.x];
    float3 b = vertices[tri.y];
    float3 c = vertices[tri.z];

    areas[tid] = 0.5f * length(cross(b - a, c - a));
};
__global__ void compute_triangle_normals(const float3* __restrict__ vertices,
                                       const int3* __restrict__ triangles,
                                       float3* __restrict__ normals,
                                       int num_triangles) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_triangles) return;

    int3 tri = triangles[tid];
    float3 a = vertices[tri.x];
    float3 b = vertices[tri.y];
    float3 c = vertices[tri.z];

    normals[tid] = normalize(cross(b - a, c - a));
};
__global__ void get_cotan_weights(float* __restrict__ edge_values, const trimesh input_mesh )
{
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(edge_idx >= input_mesh.num_edges)
        return;
    int i_from = input_mesh.edge_from[edge_idx];
    int i_to = input_mesh.edge_to[edge_idx];
    int i_other = input_mesh.edge_other[edge_idx];

    if(i_from == i_to)
    {
        return;
    };


    float3 v_from = input_mesh.vertices[i_from];
    float3 v_to = input_mesh.vertices[i_to];
    float3 v_other = input_mesh.vertices[i_other];

    

    float cotan = cotangent(v_from - v_other, v_to - v_other);
    
    atomicAdd(&edge_values[edge_idx], cotan/2.0f);
};

//next: w_ij ||u_i-u_j||^2 = w_ij (u_i^2 + v_i^2) + w_ij (u_j^2 + v_j^2) - 2 w_ij (u_i u_j + v_i v_j)
//\sum_ij  u_i (- 2 w_ij + 2 \delta_ij (\sum_k w_{ik})) u_j and same for u




