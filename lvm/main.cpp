#include "calculation.h"
#include <cmath>

int main()
{
    float w0 = 1.0f;
    float h0 = 16.0f;
    float x0 = 0.0f;
    float y0 = 0.0f;
    int num_cells = 10;
    float2* apical, *basal;
    int num_vertices = init_flat_tissue(apical, basal, num_cells, w0, h0, x0, y0, true);
    float* l_a = (float*)malloc(sizeof(float)*num_cells);
    float* l_b = (float*)malloc(sizeof(float)*num_cells);
    float* l_l = (float*)malloc(sizeof(float)*num_vertices);
    float* area = (float*)malloc(sizeof(float)*num_cells);
    float* gamma_a,* gamma_b, *gamma_l, *A0;
    init_params(gamma_a, gamma_b, gamma_l, A0, num_vertices, 1.0f, 1.0f, 1.0f, w0*h0, false, true);
    float2* apical_grads = (float2*)malloc(sizeof(float2)*num_vertices);
    float2* basal_grads = (float2*)malloc(sizeof(float2)*num_vertices);
    fill(apical_grads, num_vertices, zero2);
    fill(basal_grads, num_vertices, zero2);
    fill(A0, num_cells, w0*h0);
    
    lvm_compute_geometry(apical, basal, l_a, l_b, l_l, area, num_vertices, false);
    
    lvm_accumulate_geometry_gradients(apical, basal, l_a, l_b, l_l, area,
                                    apical_grads, basal_grads,
                                    gamma_a, gamma_b, gamma_l, A0,
                                    num_vertices, false);
    for(int i = 0; i < num_vertices; i++)
    {
        printf("Vertex %d: Apical grad = (%f, %f), Basal grad = (%f, %f)\n", i, apical_grads[i].x, apical_grads[i].y, basal_grads[i].x, basal_grads[i].y);
    }
    free(apical);
    free(basal);
    free(l_a);
    free(l_b);
    free(l_l);
    free(area);
    return 0;
};