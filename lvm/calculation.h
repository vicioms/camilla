#pragma once
#include "../generic.h"
using namespace std;

void lvm_compute_geometry(float2* apical, float2* basal, 
                            float* l_a, float* l_b, float* l_l, float* area,
                            int num_vertices)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        l_l[v_i] = length(apical[v_i] - basal[v_i]);
        if(v_i > 0)
        {
            l_a[v_i-1] = length(apical[v_i] - apical[v_i-1]);
            l_b[v_i-1] = length(basal[v_i] - basal[v_i-1]);
            area[v_i-1] = quad_area(basal[v_i-1], basal[v_i], apical[v_i], apical[v_i-1]);
        };
       
    };
};

void lvm_accumulate_geometry_gradients(float2* apical, float2* basal, 
                                    float* l_a, float* l_b, float* l_l, float* area,
                                    float2* apical_grads, float2* basal_grads,
                                    float* tension_a, float* tension_b, float* tension_l, float* A0,
                                    int num_vertices)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        float2 lateral_unit_vec = normalize(apical[v_i] - basal[v_i]);
        apical[v_i] += tension_l[v_i]*lateral_unit_vec;
        basal[v_i] -= tension_l[v_i]*lateral_unit_vec;
        if(v_i > 0)
        {
            float2 apical_unit_vec = normalize(apical[v_i] - apical[v_i-1]);
            float2 basal_unit_vec = normalize(apical[v_i] - apical[v_i-1]);
            apical[v_i] += tension_a[v_i-1]*apical_unit_vec;
            apical[v_i-1] -= tension_a[v_i-1]*apical_unit_vec;
            basal[v_i] += tension_b[v_i-1]*basal_unit_vec;
            basal[v_i-1] -= tension_b[v_i-1]*basal_unit_vec;
        };
        
        
    }
};  
