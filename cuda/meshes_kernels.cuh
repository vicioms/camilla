#pragma once
#include "meshes.cuh"

__global__ void _compute_edge_geometry(const mesh* m, float3* edge_vectors, float* edge_lengths)
{
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_edges = m->num_edges;
    if (e_idx >= num_edges)
        return;
    int2 edge = m->edge_vertices[e_idx];
    int v0 = edge.x;
    int v1 = edge.y;
    float3 p0 = m->vertices[v0];
    float3 p1 = m->vertices[v1];
    float3 edge_vector = p1 - p0;
    float edge_length = length(edge_vector);
    edge_vectors[e_idx] = edge_vector;
    edge_lengths[e_idx] = edge_length;
};


template <typename T>
__global__ void _apply_edge_incidence(const mesh* m, const T* vertex_values, T* edge_values)
{
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_edges = m->num_edges;
    if (e_idx >= num_edges)
        return;
    int2 edge = m->edge_vertices[e_idx];
    int v0 = edge.x;
    int v1 = edge.y;
    T value0 = vertex_values[v0];
    T value1 = vertex_values[v1];
    edge_values[e_idx] = value1 - value0;
};



