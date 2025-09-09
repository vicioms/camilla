#include "calculation.h"
#include <cmath>

int main()
{
    float w0 = 1.0f;
    float h0 = 16.0f;
    float x0 = 0.0f;
    float y0 = 0.0f;
    int num_cells = 10;
    float2* apical = nullptr;
    float2* basal = nullptr;
    int num_vertices = init_flat_tissue(apical, basal, num_cells, w0, h0, x0, y0, true);
    float* l_a = (float*)malloc(sizeof(float)*num_cells);
    float* l_b = (float*)malloc(sizeof(float)*num_cells);
    float* l_l = (float*)malloc(sizeof(float)*num_vertices);
    float* area = (float*)malloc(sizeof(float)*num_cells);
    lvm_compute_geometry(apical, basal, l_a, l_b, l_l, area, num_vertices, false);
    free(apical);
    free(basal);
    free(l_a);
    free(l_b);
    free(l_l);
    free(area);
    return 0;
};