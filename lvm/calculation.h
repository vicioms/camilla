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

void init_params(float*& gamma_a, float*& gamma_b, float*& gamma_l, float*& A0,
                int num_vertices, float gamma_a0, float gamma_b0, float gamma_l0, float A0_init, bool pbc,
                bool allocate)
{
    int num_cells = pbc ? num_vertices : num_vertices-1;
    if (allocate) {
        gamma_a = (float*)malloc(sizeof(float)*num_cells);
        gamma_b = (float*)malloc(sizeof(float)*num_cells);
        gamma_l = (float*)malloc(sizeof(float)*num_vertices);
        A0 = (float*)malloc(sizeof(float)*num_cells);
    }
    fill(gamma_a, num_cells, gamma_a0);
    fill(gamma_b, num_cells, gamma_b0);
    fill(gamma_l, num_vertices, gamma_l0);
    fill(A0, num_cells, A0_init);
};

void get_equilibrium_params_flat_tissue(const float w_eq, const float h_eq, const float K, const float gamma, float& gamma_l)
{
    //energy reads:
    // (K/2)*(wh-A0)^2 + 2*gamma*w + gamma_l*h
    
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
                                    float* gamma_a, float* gamma_b, float* gamma_l, float* A0, float K,
                                    int num_vertices,
                                    bool pbc)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        float2 lateral_unit_vec = normalize(apical[v_i] - basal[v_i]);
        apical_grads[v_i] += gamma_l[v_i]*lateral_unit_vec;
        basal_grads[v_i] -= gamma_l[v_i]*lateral_unit_vec;
    };
    int last_cell_index = pbc ? num_vertices : num_vertices-1;
    for(int c = 0; c < last_cell_index; c++)
    {
        int v_index = c;
        int v_index_next =  pmod(c+1, num_vertices);
        float2 apical_unit_vec = normalize(apical[v_index_next] - apical[v_index]);
        float2 basal_unit_vec = normalize(basal[v_index_next] - basal[v_index]);
        apical_grads[v_index_next] += gamma_a[v_index]*apical_unit_vec;
        apical_grads[v_index] -= gamma_a[v_index]*apical_unit_vec;
        basal_grads[v_index_next] += gamma_b[v_index]*basal_unit_vec;
        basal_grads[v_index] -= gamma_b[v_index]*basal_unit_vec;

        float area_diff = K*(area[c] - A0[c]);

        quad_area_gradients(basal[v_index], basal[v_index_next], apical[v_index_next], apical[v_index],
             basal_grads[v_index], basal_grads[v_index_next], apical_grads[v_index_next], apical_grads[v_index],
                            area_diff, true);
    };
};  
