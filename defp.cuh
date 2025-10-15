#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_math.h"

__global__ void compute_triangle_areas_and_normals_with_grads(float3* verts,
                                                int* tris,
                                                float* tri_areas,
                                                float3* tri_normals,
                                                float3* vert_grads,
                                                float k,
                                                float a0,
                                                int num_tris)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tris)
    {
        float3 v0 = verts[3*idx];
        float3 v1 = verts[3*idx+1];
        float3 v2 = verts[3*idx+2];

        float3 half_area_vector =  cross(v1 - v0, v2 - v0);

        float half_area = length(half_area_vector);
        float area = 2.0f * half_area;
        float3 normal = half_area_vector / half_area;
        tri_areas[idx] = area;
        tri_normals[idx] = normal;
        //vert_grads[3*idx] += k * (area - a0) * cross(v1 - v2, normal);
        //vert_grads[3*idx+1] += k * (area - a0) * cross(v2 - v0, normal);
        //vert_grads[3*idx+2] += k * (area - a0) * cross(v0 - v1, normal);
    }
};



class DefObjects
{
public:
    float3* verts = nullptr;
    int* tris = nullptr;
    int* tri_ids = nullptr;
    int num_verts = 0;
    int num_tris = 0;
    int num_objs = 0;
    float* tri_areas = nullptr;
    float3* tri_normals = nullptr;


};