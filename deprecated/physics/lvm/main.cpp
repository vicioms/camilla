#include "calculation.h"
#include <fstream>
#include <iostream>
#include <cmath>

int main()
{
    //int quad_order = 12;
    //float* quad_w;
    //float* quad_x;
    //int num_quad_points = get_tanh_quadrature_weights_and_points(quad_order, 1e-3, quad_w, quad_x);
    float w0 = 1.0f;
    float h0 = 16.0f;
    float K = 1.0f;
    float gamma = 3.0f;
    float r_0 = 1.0f;
    float gamma_a0 = 10.0f;
    float gamma_b0 = gamma_a0;
    float gamma_l0 =2*gamma_a0*w0/h0;
    float A0_init = w0*h0 + 2*gamma_a0/(K*h0);
    float regularization_length = 0.2f;
    float x0 = 0.0f;
    float y0 = 0.0f;
    int num_cells = 151;
    int central_cell = (num_cells-1)/2;
    int half_num_perturbed_cells = 4;
    float2* apical, *basal;
    int num_vertices = lvm_init_flat_tissue(apical, basal, num_cells, w0, h0, x0, y0, true);
    float* l_a = (float*)malloc(sizeof(float)*num_cells);
    float* l_b = (float*)malloc(sizeof(float)*num_cells);
    float* l_l = (float*)malloc(sizeof(float)*num_vertices);
    float* area = (float*)malloc(sizeof(float)*num_cells);
    float* gamma_a,* gamma_b, *gamma_l, *A0;
    lvm_init_params(gamma_a, gamma_b, gamma_l, A0, num_vertices, gamma_a0, gamma_b0, gamma_l0, A0_init, false, true);
    lvm_scale_apical_or_basal_gamma(gamma_b, central_cell-half_num_perturbed_cells, central_cell+half_num_perturbed_cells+1, num_vertices, 0.4f, false);
    lvm_scale_lateral_gamma(gamma_l, central_cell-half_num_perturbed_cells, central_cell+half_num_perturbed_cells+2, num_vertices, 2.0f, false);
    float2* apical_grads = (float2*)malloc(sizeof(float2)*num_vertices);
    float2* basal_grads = (float2*)malloc(sizeof(float2)*num_vertices);
    fill(apical_grads, num_vertices, zero2);
    fill(basal_grads, num_vertices, zero2);
    fill(A0, num_cells, w0*h0);
        
    float dt = 1e-4f;
    float F = -3.0f;
    for(int step =0; step < 100000000; step++)
    {
        lvm_compute_geometry(apical, basal, l_a, l_b, l_l, area, num_vertices, false);
        lvm_accumulate_geometry_gradients(apical, basal, l_a, l_b, l_l, area,
                                    apical_grads, basal_grads,
                                    gamma_a, gamma_b, gamma_l, A0, K, regularization_length,
                                    num_vertices, false);

        int num_pairs_of_interacting_cells = lvm_accumulate_adhesion_grads(apical, apical_grads, num_vertices, gamma, r_0, false);
        if( step % int(1/dt) == 0)
        {
            cout << dt*step << " " << num_pairs_of_interacting_cells << endl;
        }
        if(step % int(1/dt) == 0)
        {
            ofstream fout("tissue.txt");
            for(int v_i = 0; v_i < num_vertices; v_i++)
            {
                fout << basal[v_i].x << " " << basal[v_i].y << " " << apical[v_i].x << " " << apical[v_i].y << endl;
            };
            fout.close();
        }
        lvm_apply_force_boundary_conditions_flat_tissue(apical, basal,
                                apical_grads, basal_grads,
                                F,
                                true, true,
                                true, true,
                                num_vertices);
        lvm_euler_step(apical, basal,
                apical_grads, basal_grads,
                dt,
                num_vertices,
                true);

        
        
    };
    ofstream fout("tissue.txt");
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        fout << basal[v_i].x << " " << basal[v_i].y << " " << apical[v_i].x << " " << apical[v_i].y << endl;
    };
    fout.close();
    
    
    free(apical);
    free(basal);
    free(l_a);
    free(l_b);
    free(l_l);
    free(area);
    return 0;
};