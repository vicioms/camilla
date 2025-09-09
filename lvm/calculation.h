#pragma once
#include "../generic.h"
using namespace std;

int init_flat_tissue(float2*& apical, float2*& basal, 
                        int num_cells, float w0, float h0, float x0, float y0, bool allocate)
{
    int num_vertices = num_cells+1;
    if (allocate) {
        apical = (float2*)malloc(sizeof(float2)*num_vertices);
        basal = (float2*)malloc(sizeof(float2)*num_vertices);
    }
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        apical[v_i] = float2(x0+v_i*w0, y0+h0);
        basal[v_i] = float2(x0+v_i*w0, y0);
    };
    return num_vertices;
};

int init_circular_tissue(float2*& apical, float2*& basal, 
                        int num_cells, float r_inner, float r_outer, float x0, float y0, bool allocate)
{
    int num_vertices = num_cells;
    if (allocate) {
        apical = (float2*)malloc(sizeof(float2)*num_vertices);
        basal = (float2*)malloc(sizeof(float2)*num_vertices);
    }
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        float angle = 2.0f * M_PI * float(v_i) / float(num_vertices);
        apical[v_i] = float2(x0 + r_outer * cosf(angle), y0 + r_outer * sinf(angle));
        basal[v_i] = float2(x0 + r_inner * cosf(angle), y0 + r_inner * sinf(angle));
    };
    return num_vertices;
};

void lvm_compute_geometry(float2* apical, float2* basal, 
                            float* l_a, float* l_b, float* l_l, float* area,
                            int num_vertices,
                            bool pbc)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        l_l[v_i] = length(apical[v_i] - basal[v_i]);
    };
    int last_cell_index = pbc ? num_vertices : num_vertices-1;
    for(int c = 0; c < last_cell_index; c++)
    {
        int v_index = c;
        int v_index_next = pmod(c+1, num_vertices);
        l_a[c] = length(apical[v_index_next] - apical[v_index]);
        l_b[c] = length(basal[v_index_next] - basal[v_index]);
        area[c] = fabsf(quad_area_signed(basal[v_index], basal[v_index_next], apical[v_index_next], apical[v_index]));
    };
};

void lvm_accumulate_geometry_gradients(float2* apical, float2* basal, 
                                    float* l_a, float* l_b, float* l_l, float* area,
                                    float2* apical_grads, float2* basal_grads,
                                    float* tension_a, float* tension_b, float* tension_l, float* A0,
                                    int num_vertices,
                                    bool pbc)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        float2 lateral_unit_vec = normalize(apical[v_i] - basal[v_i]);
        apical[v_i] += tension_l[v_i]*lateral_unit_vec;
        basal[v_i] -= tension_l[v_i]*lateral_unit_vec;        
    };
    for(int c = 0; c < pbc ? num_vertices : num_vertices-1; c++)
    {
        int v_index = c;
        int v_index_next =  pmod(c+1, num_vertices);
        float2 apical_unit_vec = normalize(apical[v_index_next] - apical[v_index]);
        float2 basal_unit_vec = normalize(basal[v_index_next] - basal[v_index]);
        apical[v_index_next] += tension_a[v_index]*apical_unit_vec;
        apical[v_index] -= tension_a[v_index]*apical_unit_vec;
        basal[v_index_next] += tension_b[v_index]*basal_unit_vec;
        basal[v_index] -= tension_b[v_index]*basal_unit_vec;
        float area_diff = area[v_index] - A0[v_index];
        quad_area_gradients(basal[v_index], basal[v_index_next], apical[v_index_next], apical[v_index],
             basal_grads[v_index], basal_grads[v_index_next], apical_grads[v_index_next], apical_grads[v_index],
                            area_diff, true);
    };
};  
