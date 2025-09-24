#pragma once
#include "../generic.h"
using namespace std;

int lvm_init_flat_tissue(float2*& apical, float2*& basal, 
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
int lvm_init_circular_tissue(float2*& apical, float2*& basal, 
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
void lvm_init_params(float*& gamma_a, float*& gamma_b, float*& gamma_l, float*& A0,
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
void lvm_scale_apical_or_basal_gamma(float* gamma, int cell_from, int cell_to, int num_vertices, float scale, bool pbc)
{
    int num_cells = pbc ? num_vertices : num_vertices-1;
    for(int c = max(cell_from,0); c < min(cell_to, num_cells); c++)
    {
        gamma[c] *= scale;
    };
};
void lvm_scale_lateral_gamma(float* gamma, int vertex_from, int vertex_to, int num_vertices, float scale, bool pbc)
{
    for(int i = max(0, vertex_from); i < min(vertex_to, num_vertices); i++)
    {
        gamma[i] *= scale;
    };
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
                                    float* gamma_a, float* gamma_b, float* gamma_l, float* A0, float K, float regularization_length,
                                    int num_vertices,
                                    bool pbc)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        if(l_l[v_i] > regularization_length)
        {
            float2 lateral_unit_vec = normalize(apical[v_i] - basal[v_i]);
            apical_grads[v_i] += gamma_l[v_i]*lateral_unit_vec;
            basal_grads[v_i] -= gamma_l[v_i]*lateral_unit_vec;
        }
        else
        {
            float2 lateral_diff_vec = apical[v_i] - basal[v_i];
            apical_grads[v_i] += (gamma_l[v_i]/regularization_length)*lateral_diff_vec;
            basal_grads[v_i] -= (gamma_l[v_i]/regularization_length)*lateral_diff_vec;
        };
    };
    int last_cell_index = pbc ? num_vertices : num_vertices-1;
    for(int c = 0; c < last_cell_index; c++)
    {
        int v_index = c;
        int v_index_next =  pmod(c+1, num_vertices);
        if(l_a[c] > regularization_length)
        {
            float2 apical_unit_vec = normalize(apical[v_index_next] - apical[v_index]);
            apical_grads[v_index_next] += gamma_a[c]*apical_unit_vec;
            apical_grads[v_index] -= gamma_a[c]*apical_unit_vec;
        }
        else
        {
            float2 apical_diff_vec = apical[v_index_next] - apical[v_index];
            apical_grads[v_index_next] += (gamma_a[c]/regularization_length)*apical_diff_vec;
            apical_grads[v_index] -= (gamma_a[c]/regularization_length)*apical_diff_vec;
        }
        if(l_b[c] > regularization_length)
        {
            float2 basal_unit_vec = normalize(basal[v_index_next] - basal[v_index]);
            basal_grads[v_index_next] += gamma_b[c]*basal_unit_vec;
            basal_grads[v_index] -= gamma_b[c]*basal_unit_vec;
        }
        else
        {
            float2 basal_diff_vec = basal[v_index_next] - basal[v_index];
            basal_grads[v_index_next] += (gamma_b[c]/regularization_length)*basal_diff_vec;
            basal_grads[v_index] -= (gamma_b[c]/regularization_length)*basal_diff_vec;
        };

        float area_diff = K*(area[c] - A0[c]);

        quad_area_gradients(basal[v_index], basal[v_index_next], apical[v_index_next], apical[v_index],
             basal_grads[v_index], basal_grads[v_index_next], apical_grads[v_index_next], apical_grads[v_index],
                            area_diff, true);
    };
};  
void lvm_apply_force_boundary_conditions_flat_tissue(float2* apical, float2* basal,
                                float2* apical_grads, float2* basal_grads,
                                float force,
                                bool comoving_x_left, bool comoving_x_right,
                                bool fixed_y_left, bool fixed_y_right,
                                int num_vertices)
{
    float avg_left_x_grad = (apical_grads[0].x + basal_grads[0].x)*0.5f;
    float avg_right_x_grad = (apical_grads[num_vertices-1].x + basal_grads[num_vertices-1].x)*0.5f;
    //float x_c_left = (apical[0].x + basal[0].x)*0.5f;
    //float x_c_right = (apical[num_vertices-1].x + basal[num_vertices-1].x)*0.5f;
    if(comoving_x_left)
    {
        apical_grads[0].x = avg_left_x_grad + force/2.0f;
        basal_grads[0].x = avg_left_x_grad + force/2.0f;
    }
    else
    {
        apical_grads[0].x += force/2.0f;
        basal_grads[0].x += force/2.0f;
    };
    if(comoving_x_right)
    {
        apical_grads[num_vertices-1].x = avg_right_x_grad - force/2.0f;
        basal_grads[num_vertices-1].x = avg_right_x_grad - force/2.0f;
    }
    else
    {
        apical_grads[num_vertices-1].x -= force/2.0f;
        basal_grads[num_vertices-1].x -= force/2.0f;
    };
    if(fixed_y_left)
    {
        apical_grads[0].y = 0.0f;
        basal_grads[0].y = 0.0f;
    };
    if(fixed_y_right)
    {
        apical_grads[num_vertices-1].y = 0.0f;
        basal_grads[num_vertices-1].y = 0.0f;
    };
};

void lvm_euler_step(float2* apical, float2* basal,
                float2* apical_grads, float2* basal_grads,
                float dt,
                int num_vertices,
                bool zero_grads_after)
{
    for(int v_i = 0; v_i < num_vertices; v_i++)
    {
        apical[v_i] -= dt*apical_grads[v_i];
        basal[v_i] -= dt*basal_grads[v_i];
        if(zero_grads_after)
        {
            apical_grads[v_i] = zero2;
            basal_grads[v_i] = zero2;
        };
    };
};


void lvm_morse_potential(float r, float D, float a, float r0, float& potential, float& potential_radial_derivative)
{
    float exp_term = expf(-a*(r - r0));
    potential = D*exp_term*(exp_term - 2.0f);
    potential_radial_derivative = -2.0f*D*a*exp_term*(exp_term - 1.0f);
};
bool lvm_lj_potential(float r, float gamma,  float r0, float lj_power, float& potential, float& potential_radial_derivative)
{
    float factor = powf(r0/r, lj_power);
    potential = gamma*factor*(factor - 2.0f);
    potential_radial_derivative = gamma*(2*lj_power/r)*(factor-factor*factor);
    return true;
};
inline bool lvm_harmonic_potential(
    float r,
    float Ec,         // adhesion/“cohesion” strength (d_ec)
    float r0,         // length scale (like radSum)
    float l1, float l2,
    float& potential,
    float& potential_radial_derivative)
{
    // guard
    const float rc  = (1.0f + l2) * r0;
    if (r >= rc || l2 <= l1) {
        potential = 0.0f;
        potential_radial_derivative = 0.0f;
        return false;
    }

    const float x   = 1.0f - r / r0;          // overlap
    const float rin = (1.0f + l1) * r0;       // inner join

    if (r < rin) {
        // Inner quadratic well
        // V = 0.25*Ec*(x^2 - l1*l2)
        potential = 0.25f * Ec * (x * x - l1 * l2);
        // dV/dr = 0.5*Ec*x * d(x)/dr,  d(x)/dr = -1/r0
        potential_radial_derivative = -(0.5f * Ec / r0) * x;
    } else {
        // Outer taper
        // V = -0.25 * (Ec*l1/(l2 - l1)) * (x + l2)^2
        const float A = Ec * l1 / (l2 - l1);
        potential = -0.25f * A * (x + l2) * (x + l2);
        // dV/dr = -0.5*A*(x+l2) * d(x)/dr = +0.5*A*(x+l2)/r0
        potential_radial_derivative =  (0.5f * A / r0) * (x + l2);
    }
    return true;
};

inline bool lvm_angular_potential(float2 r_vec, float2 p_vec, float2 q_vec, float k_theta,  float r_cutoff, float& potential, float& potential_derivative_r, float2& potential_derivative_p, float2& potential_derivative_q)
{
    float r = length(r_vec);
    if(r>r_cutoff)
    {
        potential = 0.0f;
        potential_derivative_r = 0.0f;
        return false;
    }
    float p_len = length(p_vec);
    float q_len = length(q_vec);
    float2 u = p_vec/p_len;
    float2 v = q_vec/q_len;
    float cos_theta = dot(u,v);
    float s = 1 - r/r_cutoff;
    float w = s*s*(3 - 2*s);
    float w_der_r = -6.0f*s*(1-s)/r_cutoff;
    potential = 0.5f*k_theta*(1.0f - cos_theta*cos_theta)*w;
    potential_derivative_r = 0.5f*k_theta*(1.0f - cos_theta*cos_theta)*w_der_r;
    potential_derivative_p = -2*k_theta*w*cos_theta*(v-cos_theta*u)/p_len;
    potential_derivative_q = -2*k_theta*w*cos_theta*(u-cos_theta*v)/q_len;
    return true;
};

bool lvm_point_segment_interaction(const float2 p, const float2 p1, const float2 p2, float2& g_p, float2& g1, float2& g2, float adhesion_strength, float r_c, bool accumulate)
{
    if(!accumulate)
    {
        g_p = zero2;
        g1 = zero2;
        g2 = zero2;
    }
    if(p == p1 || p == p2)
        return false;
    float2 seg_vec = p2 - p1;
    float2 p1_to_p = p - p1;
    float t = dot(p1_to_p, seg_vec)/dot(seg_vec,seg_vec);
    t = fmaxf(fminf(t, 1.0f), 0.0f);
    float2 closest_point = p1 + t*seg_vec;
    float2 r_vec = p - closest_point;
    float r = length(r_vec);
    float2 r_hat = r_vec/r;
    float potential_value = 0.0f;
    float potential_value_der = 0.0f;
    if(!lvm_harmonic_potential(r, adhesion_strength, r_c, 2.0f, 4.0f, potential_value, potential_value_der))
        return false;
    g_p += potential_value_der*r_hat;
    g1 -= potential_value_der*(1.0f - t)*r_hat;
    g2 -= potential_value_der*t*r_hat;
    return true;
};

inline bool lvm_edge_edge_grads(
    const float2 p1, const float2 p2,
    const float2 q1, const float2 q2,
    float strength, float r_0,
    float2& gp1, float2& gp2,
    float2& gq1, float2& gq2)
{
    gp1 = gp2 = gq1 = gq2 = make_float2(0.f,0.f);

    float s, t; float2 r_vec;
    get_closest_points_segments(p1, p2, q1, q2, s, t, r_vec);
    const float r2 = dot(r_vec, r_vec);
    if (r2 <= 1e-24f) return false;            // coincident -> no direction
    const float r = sqrtf(r2);

    const float2 r_hat = (1.0f / r) * r_vec;

    float V, dVdr;
    bool interacting = lvm_harmonic_potential(r, strength, r_0, 2.0f, 4.0f, V, dVdr);
    if(!interacting)
    {
        return false;
    }
   

    const float wP1 = 1.0f - s, wP2 = s;
    const float wQ1 = 1.0f - t, wQ2 = t;

    const float2 gP =  dVdr * r_hat;             // force on P segment (acts to pull towards Q)
    const float2 gQ = -dVdr * r_hat;             // equal and opposite on Q segment

    gp1 += wP1 * gP;
    gp2 += wP2 * gP;
    gq1 += wQ1 * gQ;
    gq2 += wQ2 * gQ;

    float2 potential_derivative_p, potential_derivative_q;
    float potential, potential_derivative_r;
    lvm_angular_potential(r_vec, p2 - p1, q2 - q1, strength, 5.0f*r_0, potential, potential_derivative_r, potential_derivative_p, potential_derivative_q);
    if(interacting)
    {
        gp1 += -1.0f*potential_derivative_p+wP1*potential_derivative_r*r_hat;
        gp2 += potential_derivative_p+wP2*potential_derivative_r*r_hat;
        gq1 += -1.0f*potential_derivative_q+wQ1*potential_derivative_r*r_hat;
        gq2 += potential_derivative_q+wQ2*potential_derivative_r*r_hat;
    }
    return true;
}

int lvm_accumulate_adhesion_grads(float2* apical, float2* apical_grads, int num_vertices, float adhesion_strength, float r_c, bool pbc)
{
    int num_pairs_of_interactin_cells = 0;
    int num_cells = pbc ? num_vertices : num_vertices-1;

    for(int c = 0; c < num_cells; c++)
    {
        int i_1_in = c;
        int i_1_fin = pmod(c+1, num_vertices);
        float2 midpoint_c = 0.5f*(apical[i_1_in] + apical[i_1_fin]);
        
        float2 dir_c = normalize(apical[i_1_fin] - apical[i_1_in]);
        //if(dir_c.y > -1e-3)
        //    continue;
        float2 n_c = make_float2(-dir_c.y, dir_c.x);

        for(int cp = c+1; cp < num_cells; cp++)
        {

            int i_2_in = cp;
            int i_2_fin = pmod(cp+1, num_vertices);

            float2 midpoint_cp = 0.5f*(apical[i_2_in] + apical[i_2_fin]);

            float2 midpoint_vec = midpoint_cp - midpoint_c;
            float2 dir_cp = normalize(apical[i_2_fin] - apical[i_2_in]);
            float2 n_cp = make_float2(-dir_cp.y, dir_cp.x);
            if(dot(n_cp, n_c) > 0.0f)
            {
                continue;
            }
            //float normals_dot = dot(n_c, n_cp);
            //if(dir_cp.y < 1e-3)
            //    continue;
            
            if(dot(midpoint_vec, n_c) < 0.0f || dot(midpoint_vec, n_cp) > 0.0f)
                continue;
            
            float2 g1_in, g2_in, g1_final, g2_final;
            g1_in = zero2;
            g2_in = zero2;
            g1_final = zero2;
            g2_final = zero2;
            bool interacting = lvm_edge_edge_grads(
                apical[i_1_in], apical[i_1_fin],
                apical[i_2_in], apical[i_2_fin],
                adhesion_strength, r_c,
                g1_in, g1_final, g2_in, g2_final);

            
            if(interacting)
            {
                num_pairs_of_interactin_cells += 1;
                apical_grads[i_1_in] += g1_in;
                apical_grads[i_1_fin] += g1_final;
                apical_grads[i_2_in] += g2_in;
                apical_grads[i_2_fin] += g2_final;   
            }
            //if(i_1_in != i_2_in && i_1_in != i_2_fin)
            //{
            //    interacting |= lvm_point_segment_interaction(apical[i_1_in], apical[i_2_in], apical[i_2_fin],
            //        g1_in, g2_in, g2_final, adhesion_strength, r_c, true);
            //}
            //if(i_1_fin != i_2_in && i_1_fin != i_2_fin)
            //{
            //    interacting |= lvm_point_segment_interaction(apical[i_1_fin], apical[i_2_in], apical[i_2_fin],
            //        g1_final, g2_in, g2_final, adhesion_strength, r_c, true);
            //}
            //if(i_2_in != i_1_in && i_2_in != i_1_fin)
            //{
            //    interacting |= lvm_point_segment_interaction(apical[i_2_in], apical[i_1_in], apical[i_1_fin],
            //        g2_in, g1_in, g1_final, adhesion_strength, r_c, true);
            //}
            //if(i_2_fin != i_1_in && i_2_fin != i_1_fin)
            //{
            //    interacting |= lvm_point_segment_interaction(apical[i_2_fin], apical[i_1_in], apical[i_1_fin],
            //        g2_final, g1_in, g1_final, adhesion_strength, r_c, true);
            //}
            //if(interacting)
            //{
            //    num_pairs_of_interactin_cells += 1;
            //    apical_grads[i_1_in] += g1_in;
            //    apical_grads[i_2_in] += g2_in;
            //    apical_grads[i_1_fin] += g1_final;
            //    apical_grads[i_2_fin] += g2_final;   
            //}

            //if(lvm_adhesion_grads_endpoints(
            //    apical[i_1_in], apical[i_2_in],
            //                    apical[i_1_fin], apical[i_2_fin],
            //                    adhesion_strength, r_c,
            //                    g1_in, g2_in, g1_final, g2_final))
            //{
            //    num_pairs_of_interactin_cells += 1;
            //    apical_grads[i_1_in] += g1_in;
            //    apical_grads[i_2_in] += g2_in;
            //    apical_grads[i_1_fin] += g1_final;
            //    apical_grads[i_2_fin] += g2_final;   
            //}
            
             
        }
            
    };
    return num_pairs_of_interactin_cells;
};

