#include <cmath>
#include "hmath.h"
#include <iostream>
#include <fstream>

using namespace std;

struct lvm
{
    int n_cells = 0;
    real l0 = 0.0;
    real K = 0.0;
    real A0 = 0.0;
    real* gamma_a = nullptr;
    real* gamma_b = nullptr;
    real* gamma_l = nullptr;
    real K_bd = 0.0;


    vec2* apical = nullptr;
    vec2* basal = nullptr;
    vec2 bd_left = vec2(0.0, 0.0);
    vec2 bd_right = vec2(0.0, 0.0);

    real* length_a = nullptr;
    real* length_b = nullptr;
    real* length_l = nullptr;
    real* area = nullptr;


    vec2* apical_tension_grads = nullptr;
    vec2* basal_tension_grads = nullptr;
    vec2* apical_area_grads = nullptr;
    vec2* basal_area_grads = nullptr;
    vec2* apical_interaction_grads = nullptr;
    vec2* basal_interaction_grads = nullptr;
    vec2* apical_residual_grads = nullptr;
    vec2* basal_residual_grads = nullptr;
    vec2 bd_left_grad = vec2(0.0, 0.0);
    vec2 bd_right_grad = vec2(0.0, 0.0);

    int num_gl = 0; 
    real* gl_points = nullptr;
    real* gl_weights = nullptr;

    lvm() = default;

    explicit lvm(int n_cells_, real l0_, real K_, real h0, real w0, real gamma_a0, real gamma_b0, real K_bd_, int num_gauss_legendre_points_)
    {

        //params init
        n_cells = n_cells_;
        l0 = l0_;
        K = K_;
        A0 = w0 * h0 + (gamma_a0 + gamma_b0) / (2 * K_ * h0);
        real gamma_l0 = w0 * (gamma_a0 + gamma_b0) / (2 * h0);
        gamma_a = new real[n_cells];
        gamma_b = new real[n_cells];
        gamma_l = new real[n_cells+1];
        K_bd = K_bd_;
        for(int i = 0; i < n_cells+1; i++)
        {
            gamma_l[i] = gamma_l0;
            if(i < n_cells)
            {
                gamma_a[i] = gamma_a0;
                gamma_b[i] = gamma_b0;
            }
        };
        
        //state init
        apical = new vec2[n_cells+1];
        basal = new vec2[n_cells+1];
        length_a = new real[n_cells];
        length_b = new real[n_cells];
        length_l = new real[n_cells+1];
        area = new real[n_cells];

        for(int i = 0; i < n_cells+1; i++)
        {
            apical[i] = vec2(i*w0, h0);
            basal[i] = vec2(i*w0, 0.0);
        }
        bd_left = vec2(0.0, h0);
        bd_right = vec2(n_cells*w0, h0);

       

        
        //grads init
        apical_tension_grads = new vec2[n_cells+1];
        basal_tension_grads = new vec2[n_cells+1];
        apical_area_grads = new vec2[n_cells+1];
        basal_area_grads = new vec2[n_cells+1];
        apical_interaction_grads = new vec2[n_cells+1];
        basal_interaction_grads = new vec2[n_cells+1];
        apical_residual_grads = new vec2[n_cells+1];
        basal_residual_grads = new vec2[n_cells+1];


        //gauss-legendre init
        num_gl = num_gauss_legendre_points_; 
        GaussLegendreQuadrature quad = create_gauss_legendre(num_gl);
        gl_points = new real[num_gl];
        gl_weights = new real[num_gl];
        for(int i = 0; i < num_gl; i++)        {
            gl_points[i] = gauss_legendre_point(quad.x[i], 0.0f, 1.0f);
            gl_weights[i] = gauss_legendre_weight(quad.w[i], 0.0f, 1.0f);
        }

        
        //reset grads
        zero_grad();
    }

    void zero_grad();
    void tension_step();
    void area_step();
    void boundary_step(real F, real y_anchor, bool use_y_anchor);
    void step(real dt, bool zero_grad_after_step);
    void set_smooth_gamma_ab(real gamma, real delta_gamma, real scale)
    {
        for(int i = 0; i < n_cells; i++)
        {
            real factor = std::abs(std::tanh((i - n_cells/2.0) / scale));
            gamma_a[i] = gamma + delta_gamma*factor;
            gamma_b[i] = gamma - delta_gamma*factor;
        }
    }
    void scale_gamma(float scale, int cell_from, int cell_to, bool scale_apical)
    {
        for(int i = cell_from; i < min(cell_to, n_cells); i++)
        {
            if(scale_apical)
                gamma_a[i] *= scale;
            else
                gamma_b[i] *= scale;
        }
    }
    void scale_gamma_l(float scale, int edge_from, int edge_to)
    {
        for(int i = edge_from; i < min(edge_to, n_cells+1); i++)
        {
            gamma_l[i] *= scale;
        }
    }

    ~lvm()
    {
        delete[] apical;
        delete[] basal;
        delete[] length_a;
        delete[] length_b;
        delete[] length_l;
        delete[] area;
        delete[] gamma_a;
        delete[] gamma_b;
        delete[] gamma_l;
        delete[] apical_tension_grads;
        delete[] basal_tension_grads;
        delete[] apical_area_grads;
        delete[] basal_area_grads;
        delete[] apical_interaction_grads;
        delete[] basal_interaction_grads;
        delete[] apical_residual_grads;
        delete[] basal_residual_grads;
        delete[] gl_points;
        delete[] gl_weights;
    }
};

vec2 get_regularized_length_grad(vec2 edge, real length, real regularization_length)
{
    vec2 grad;
    if(length > regularization_length)
    {
        grad.x = edge.x/length;
        grad.y = edge.y/length;
    }
    else
    {
        grad.x = 2*edge.x/regularization_length;
        grad.y = 2*edge.y/regularization_length;
    };
    return grad;
}

real tri_area(vec2 a, vec2 b, vec2 c)
{
    return 0.5 * cross(b - a, c - a);
};
real quad_area(vec2 a, vec2 b, vec2 c, vec2 d)
{
    real area = cross(a, b);
    area += cross(b, c);
    area += cross(c, d);
    area += cross(d, a);
    area *= 0.5;
    return area;
}



void segment_to_segment_interaction(
    real strength,
    real a,
    real r_0,
    real r_cutoff,
    vec2 p1,
    vec2 p2,
    vec2 q1,
    vec2 q2,
    int num_gl,
    const real* gl_points,
    const real* gl_weights,
    vec2& grad_p1,
    vec2& grad_p2,
    vec2& grad_q1,
    vec2& grad_q2,
    real eps = R(1e-12)
)
{
    const vec2 delta_i = p2 - p1;
    const vec2 delta_j = q2 - q1;

    const real L_i = length(delta_i);
    const real L_j = length(delta_j);

    if (L_i <= eps || L_j <= eps)
        return;

    const vec2 e_i = delta_i / L_i;
    const vec2 e_j = delta_j / L_j;

    const real exp_cut = std::exp(-a * (r_cutoff - r_0));
    const real exp2_cut = exp_cut * exp_cut;

    const real f_cut =
        strength * (exp2_cut - R(2.0) * exp_cut);

    const real f_der_cut =
        R(2.0) * strength * a * (exp_cut - exp2_cut);

    const real r_cutoff2 = r_cutoff * r_cutoff;

    real I = R(0.0);

    vec2 dI_p1 = zero_vec2;
    vec2 dI_p2 = zero_vec2;
    vec2 dI_q1 = zero_vec2;
    vec2 dI_q2 = zero_vec2;

    for (int idx = 0; idx < num_gl; ++idx)
    {
        const real t_i = gl_points[idx];
        const real w_i = gl_weights[idx];

        const vec2 x_i = p1 + t_i * delta_i;

        for (int jdx = 0; jdx < num_gl; ++jdx)
        {
            const real t_j = gl_points[jdx];
            const real w_j = gl_weights[jdx];

            const vec2 x_j = q1 + t_j * delta_j;
            const vec2 rvec = x_j - x_i;

            // local facing mask
            if (cross(delta_i, rvec) <= R(0.0))
                continue;

            if (cross(delta_j, -rvec) <= R(0.0))
                continue;

            const real dr2 = dot(rvec, rvec);

            if (dr2 > r_cutoff2)
                continue;

            const real dr = std::sqrt(dr2);

            const real exp_val = std::exp(-a * (dr - r_0));
            const real exp2_val = exp_val * exp_val;

            const real f0 =
                strength * (exp2_val - R(2.0) * exp_val);

            const real f0_der =
                R(2.0) * strength * a * (exp_val - exp2_val);

            // shifted-force cutoff
            const real f =
                f0 - f_cut - f_der_cut * (dr - r_cutoff);

            const real f_der =
                f0_der - f_der_cut;

            const vec2 grad_r = (f_der / dr) * rvec;

            const real ww = w_i * w_j;

            I += ww * f;

            dI_p1 += ww * (-(R(1.0) - t_i) * grad_r);
            dI_p2 += ww * (-t_i * grad_r);

            dI_q1 += ww * ((R(1.0) - t_j) * grad_r);
            dI_q2 += ww * (t_j * grad_r);
        }
    }

    const real LL = L_i * L_j;

    // E = L_i L_j I
    //
    // dL_i/dp1 = -e_i
    // dL_i/dp2 =  e_i
    // dL_j/dq1 = -e_j
    // dL_j/dq2 =  e_j

    grad_p1 += LL * dI_p1 - e_i * L_j * I;
    grad_p2 += LL * dI_p2 + e_i * L_j * I;

    grad_q1 += LL * dI_q1 - e_j * L_i * I;
    grad_q2 += LL * dI_q2 + e_j * L_i * I;
}

void multipole_expansion_interaction(real a, int p, vec2 p1, vec2 p2, vec2 q1, vec2 q2, vec2& grad_p1, vec2& grad_p2, vec2& grad_q1, vec2& grad_q2)
{
    
}
/*
void segment_to_segment_interaction(
    real strength,
    real a,
    real r_0,
    real r_cutoff,
    vec2 p1,
    vec2 p2,
    vec2 q1,
    vec2 q2,
    int num_gl,
    const real* gl_points,
    const real* gl_weights,
    vec2& grad_p1,
    vec2& grad_p2,
    vec2& grad_q1,
    vec2& grad_q2,
    real eps = std::numeric_limits<real>::epsilon() * R(100.0)
)
{
    const vec2 delta_i = p2 - p1;
    const vec2 delta_j = q2 - q1;

    const real L_i = length(delta_i);
    const real L_j = length(delta_j);

    if (L_i <= R(0.0) || L_j <= R(0.0))
        return;

    const vec2 e_i = delta_i / L_i;
    const vec2 e_j = delta_j / L_j;

    const real exp_cut = std::exp(-a * (r_cutoff - r_0));

    const real f_cut =
        strength * (R(1.0) - exp_cut) * (R(1.0) - exp_cut);

    const real f_der_cut =
        R(2.0) * strength * a * (R(1.0) - exp_cut) * exp_cut;

    for (int idx = 0; idx < num_gl; ++idx)
    {
        const real t_i = gl_points[idx];
        const real w_i = gl_weights[idx];

        const vec2 x_i = p1 + t_i * delta_i;

        for (int jdx = 0; jdx < num_gl; ++jdx)
        {
            const real t_j = gl_points[jdx];
            const real w_j = gl_weights[jdx];

            const vec2 x_j = q1 + t_j * delta_j;

            const vec2 rvec = x_j - x_i;
            const real dr2 = dot(rvec, rvec);

            if (dr2 <= eps)
                continue;

            if (cross(delta_i, rvec) <= 0 || cross(delta_j, -rvec) <= 0)
                continue;

            const real dr = std::sqrt(dr2);

            if (dr > r_cutoff)
                continue;

            const real exp_val = std::exp(-a * (dr - r_0));

            const real f0 =
                strength * (R(1.0) - exp_val) * (R(1.0) - exp_val);

            const real f0_der =
                R(2.0) * strength * a * (R(1.0) - exp_val) * exp_val;

            // Shifted-force cutoff:
            // f(r_cutoff) = 0 and f'(r_cutoff) = 0
            const real f =
                f0 - f_cut - f_der_cut * (dr - r_cutoff);

            const real f_der =
                f0_der - f_der_cut;

            const vec2 grad_r = (f_der / dr) * rvec;

            const real ww = w_i * w_j;

            grad_p1 += ww * (
                -e_i * L_j * f
                - (R(1.0) - t_i) * L_i * L_j * grad_r
            );

            grad_p2 += ww * (
                 e_i * L_j * f
                - t_i * L_i * L_j * grad_r
            );

            grad_q1 += ww * (
                -e_j * L_i * f
                + (R(1.0) - t_j) * L_i * L_j * grad_r
            );

            grad_q2 += ww * (
                 e_j * L_i * f
                + t_j * L_i * L_j * grad_r
            );
        }
    }
}
*/

void lvm::zero_grad()
{
    for(int i = 0; i < n_cells+1; i++)
    {
        apical_tension_grads[i] = vec2(0.0, 0.0);
        basal_tension_grads[i] = vec2(0.0, 0.0);
        apical_area_grads[i] = vec2(0.0, 0.0);
        basal_area_grads[i] = vec2(0.0, 0.0);
        apical_interaction_grads[i] = vec2(0.0, 0.0);
        basal_interaction_grads[i] = vec2(0.0, 0.0);
        apical_residual_grads[i] = vec2(0.0, 0.0);
        basal_residual_grads[i] = vec2(0.0, 0.0);
    };
    bd_left_grad = vec2(0.0, 0.0);
    bd_right_grad = vec2(0.0, 0.0);
};

void lvm::tension_step()
{
    for(int i = 0; i < n_cells+1; i++)
    {
        vec2 lateral_edge = apical[i] - basal[i];
        real lateral_length2 = dot(lateral_edge, lateral_edge);
        real lateral_length = sqrt(lateral_length2);
        vec2 lateral_length_grad = get_regularized_length_grad(lateral_edge, lateral_length, l0);
        apical_tension_grads[i] += gamma_l[i]*lateral_length_grad;
        basal_tension_grads[i] -= gamma_l[i]*lateral_length_grad;
        if(i == n_cells)
            break;
        vec2 apical_edge = apical[i+1] - apical[i];
        real apical_length2 = dot(apical_edge, apical_edge);
        real apical_length = sqrt(apical_length2);
        vec2 basal_edge = basal[i+1] - basal[i];
        real basal_length2 = dot(basal_edge, basal_edge);
        real basal_length = sqrt(basal_length2);
        vec2 apical_length_grad = get_regularized_length_grad(apical_edge, apical_length, l0);
        vec2 basal_length_grad = get_regularized_length_grad(basal_edge, basal_length, l0);
        apical_tension_grads[i+1] += gamma_a[i]*apical_length_grad;
        apical_tension_grads[i] -= gamma_a[i]*apical_length_grad;
        basal_tension_grads[i+1] += gamma_b[i]*basal_length_grad;
        basal_tension_grads[i] -= gamma_b[i]*basal_length_grad;
    };
};

void lvm::area_step()
{
    for(int c = 0; c < n_cells; c++)
    {
        vec2 apical_edge = apical[c+1] - apical[c];
        vec2 basal_edge = basal[c+1] - basal[c];
        vec2 lateral_edge = apical[c] - basal[c];
        vec2 next_lateral_edge = apical[c+1] - basal[c+1];

        vec2 diagonal = apical[c+1] - basal[c];

        real area = 0.5 * cross(basal_edge, diagonal);
        area += 0.5 * cross(diagonal, lateral_edge);

        vec2 area_grad_apical_c =
            0.5 * perp(diagonal);

        vec2 area_grad_apical_cp1 =
            0.5 * (perp(basal_edge) - perp(lateral_edge));

        vec2 area_grad_basal_c =
            0.5 * (perp(lateral_edge) - perp(basal_edge));

        vec2 area_grad_basal_cp1 =
            -0.5 * perp(diagonal);

        real prefactor = K * (area - A0);

        apical_area_grads[c] += prefactor * area_grad_apical_c;;
        apical_area_grads[c+1] += prefactor * area_grad_apical_cp1;
        basal_area_grads[c] += prefactor * area_grad_basal_c;
        basal_area_grads[c+1] += prefactor * area_grad_basal_cp1;
    }
}

void lvm::boundary_step(real F, real y_anchor, bool use_y_anchor)
{
    bd_left_grad.x = K_bd * (bd_left.x - apical[0].x) + K_bd*(bd_left.x - basal[0].x) + F;
    bd_left_grad.y = 0.0;
    bd_right_grad.x = K_bd * (bd_right.x - apical[n_cells].x) + K_bd*(bd_right.x - basal[n_cells].x) - F;
    bd_right_grad.y = 0.0;
    apical_residual_grads[0].x += K_bd * (apical[0].x - bd_left.x);
    basal_residual_grads[0].x += K_bd * (basal[0].x - bd_left.x);
    apical_residual_grads[n_cells].x += K_bd * (apical[n_cells].x - bd_right.x);
    basal_residual_grads[n_cells].x += K_bd * (basal[n_cells].x - bd_right.x);
    if(use_y_anchor)
    {
        apical_residual_grads[0].y += K_bd * (apical[0].y - y_anchor);
        apical_residual_grads[n_cells].y += K_bd * (apical[n_cells].y - y_anchor);
        basal_residual_grads[0].y += K_bd * basal[0].y;
        basal_residual_grads[n_cells].y += K_bd * basal[n_cells].y;
    }
};

void lvm::step(real dt, bool zero_grad_after_step)
{
    for(int i = 0; i < n_cells+1; i++)
    {
        apical[i] -= dt * (apical_tension_grads[i] + apical_area_grads[i] + apical_interaction_grads[i] + apical_residual_grads[i]);
        basal[i] -= dt * (basal_tension_grads[i] + basal_area_grads[i] + basal_interaction_grads[i] + basal_residual_grads[i]);
    }
    bd_left -= dt * bd_left_grad;
    bd_right -= dt * bd_right_grad;
    if(zero_grad_after_step)
        zero_grad();
};

inline real point_segment_distance2(vec2 x, vec2 a, vec2 b)
{
    vec2 ab = b - a;
    real ab2 = dot(ab, ab);

    if (ab2 <= R(1e-24))
        return dot(x - a, x - a);

    real t = dot(x - a, ab) / ab2;
    t = std::max(R(0.0), std::min(R(1.0), t));

    vec2 closest = a + t * ab;
    vec2 d = x - closest;

    return dot(d, d);
}

inline real segment_segment_distance2(vec2 p1, vec2 p2, vec2 q1, vec2 q2)
{
    real d1 = point_segment_distance2(p1, q1, q2);
    real d2 = point_segment_distance2(p2, q1, q2);
    real d3 = point_segment_distance2(q1, p1, p2);
    real d4 = point_segment_distance2(q2, p1, p2);

    return std::min(std::min(d1, d2), std::min(d3, d4));
}

void find_interacting_pairs(
    vec2* vertices,
    int n_edges,
    real cutoff_radius,
    std::vector<std::pair<int, int>>& interacting_pairs
)
{
    real cutoff_radius2 = cutoff_radius * cutoff_radius;

    interacting_pairs.clear();

    for (int c = 0; c < n_edges; ++c)
    {
        vec2 p1 = vertices[c];
        vec2 p2 = vertices[c + 1];
        for (int c_other = c + 1; c_other < n_edges; ++c_other)
        {
            vec2 q1 = vertices[c_other];
            vec2 q2 = vertices[c_other + 1];

            // global orientation precheck
            real tri_1_area = tri_area(p1, p2, q1);
            if (tri_1_area <= R(0.0))
                continue;

            real tri_2_area = tri_area(q1, q2, p1);
            if (tri_2_area <= R(0.0))
                continue;

            // distance precheck
            real d2 = segment_segment_distance2(p1, p2, q1, q2);

            if (d2 > cutoff_radius2)
                continue;

            interacting_pairs.push_back({c, c_other});
        }
    }
}

void segment_to_segment_interaction(real strength, real a, real r_0, real r_cutoff, vec2* vertices, int n_edges, vec2* gradients,  int num_gl, const real* gl_points, const real* gl_weights, real eps = R(0.0))
{
   for(int i = 0; i < n_edges; i ++)
   {
        vec2 p1 = vertices[i];
        vec2 p2 = vertices[i+1];
        for(int j = i+1; j < n_edges; j++)
        {
            vec2 q1 = vertices[j];
            vec2 q2 = vertices[j+1];

            // first gate, check the sign of the triangle formed by the vertices
            // both must have positive area
            if(tri_area(p1, p2, q1) <= R(0.0))
                continue;
            if(tri_area(q1, q2, p1) <= R(0.0))
                continue;

            
            vec2 grad_p1 = vec2(0.0, 0.0);
            vec2 grad_p2 = vec2(0.0, 0.0);
            vec2 grad_q1 = vec2(0.0, 0.0);
            vec2 grad_q2 = vec2(0.0, 0.0);
            
        }
   }
};
void segment_to_segment_geom_interaction(real strength, real d_0_orth, real d_0_parallel, real d_0_orth_max, vec2* vertices, int n_edges, vec2* gradients, real eps = R(1e-12))
{
    if (d_0_orth <= eps || d_0_parallel <= eps)
        return;

    // Safety floor for normal LJ.
    // Prevents singularity when d_orth <= 0.
    const real d_min = R(0.15) * d_0_orth;

    auto psi = [&](real d)
    {
        d = std::max(d, d_min);

        real u = d_0_orth / d;
        real u6 = std::pow(u, R(6.0));

        return u6 * u6 - R(2.0) * u6;
    };

    auto psi_der = [&](real d)
    {
        d = std::max(d, d_min);

        real u = d_0_orth / d;
        real u6 = std::pow(u, R(6.0));
        real u12 = u6 * u6;

        // d/dd [u^12 - 2u^6]
        return (R(12.0) / d) * (u6 - u12);
    };

    const real psi_cut = psi(d_0_orth_max);
    const real psi_der_cut = psi_der(d_0_orth_max);

    for (int i = 0; i < n_edges; ++i)
    {
        vec2 p1 = vertices[i];
        vec2 p2 = vertices[i + 1];

        vec2 c_i = R(0.5) * (p1 + p2);
        vec2 edge_i = p2 - p1;

        real L_i = length(edge_i);
        if (L_i <= eps)
            continue;

        vec2 n_i = perp(edge_i) / L_i;

        for (int j = i + 2; j < n_edges; ++j)
        {
            vec2 q1 = vertices[j];
            vec2 q2 = vertices[j + 1];

            vec2 c_j = R(0.5) * (q1 + q2);
            vec2 edge_j = q2 - q1;

            real L_j = length(edge_j);
            if (L_j <= eps)
                continue;

            vec2 n_j = perp(edge_j) / L_j;

            vec2 dc = c_j - c_i;

            // Active-side facing test.
            if (dot(n_i, dc) <= R(0.0))
                continue;

            if (dot(n_j, -dc) <= R(0.0))
                continue;

            vec2 n_diff = n_i - n_j;
            real n_diff_len = length(n_diff);

            if (n_diff_len <= eps)
                continue;

            vec2 n_ij = n_diff / n_diff_len;

            real d_orth = dot(n_ij, dc);

            // Do NOT skip d_orth < 0.
            // Negative d_orth means penetration, and LJ must repel it.
            if (d_orth > d_0_orth_max)
                continue;

            real d_eval = std::max(d_orth, d_min);

            vec2 d_parallel_vec = dc - d_orth * n_ij;
            real d_parallel = length(d_parallel_vec);

            if (d_parallel > d_0_parallel)
                continue;

            real w_parallel = R(1.0) - d_parallel / d_0_parallel;

            real psi_val =
                psi(d_eval)
                - psi_cut
                - psi_der_cut * (d_eval - d_0_orth_max);

            real psi_der_val =
                psi_der(d_eval)
                - psi_der_cut;

            real pref_orth =
                strength * w_parallel * psi_der_val;

            real pref_parallel =
                -strength * psi_val / d_0_parallel;

            // -------------------------------------------------------------
            // grad d_orth
            // -------------------------------------------------------------

            vec2 h_orth =
                (dc - dot(dc, n_ij) * n_ij) / n_diff_len;

            vec2 h_i_orth = h_orth - dot(h_orth, n_i) * n_i;
            vec2 h_j_orth = h_orth - dot(h_orth, n_j) * n_j;

            vec2 g_edge_i_orth = -perp(h_i_orth) / L_i;
            vec2 g_edge_j_orth =  perp(h_j_orth) / L_j;

            vec2 gd_p1 =
                -R(0.5) * n_ij
                - g_edge_i_orth;

            vec2 gd_p2 =
                -R(0.5) * n_ij
                + g_edge_i_orth;

            vec2 gd_q1 =
                 R(0.5) * n_ij
                - g_edge_j_orth;

            vec2 gd_q2 =
                 R(0.5) * n_ij
                + g_edge_j_orth;

            // -------------------------------------------------------------
            // grad d_parallel
            // -------------------------------------------------------------

            vec2 gp_p1 = zero_vec2;
            vec2 gp_p2 = zero_vec2;
            vec2 gp_q1 = zero_vec2;
            vec2 gp_q2 = zero_vec2;

            if (d_parallel > eps)
            {
                vec2 m = d_parallel_vec / d_parallel;

                gp_p1 += -R(0.5) * m;
                gp_p2 += -R(0.5) * m;

                gp_q1 +=  R(0.5) * m;
                gp_q2 +=  R(0.5) * m;

                vec2 h_parallel =
                    (-d_orth / n_diff_len) * m;

                vec2 h_i_parallel =
                    h_parallel - dot(h_parallel, n_i) * n_i;

                vec2 h_j_parallel =
                    h_parallel - dot(h_parallel, n_j) * n_j;

                vec2 g_edge_i_parallel =
                    -perp(h_i_parallel) / L_i;

                vec2 g_edge_j_parallel =
                     perp(h_j_parallel) / L_j;

                gp_p1 += -g_edge_i_parallel;
                gp_p2 +=  g_edge_i_parallel;

                gp_q1 += -g_edge_j_parallel;
                gp_q2 +=  g_edge_j_parallel;
            }

            gradients[i] +=
                pref_orth * gd_p1
                + pref_parallel * gp_p1;

            gradients[i + 1] +=
                pref_orth * gd_p2
                + pref_parallel * gp_p2;

            gradients[j] +=
                pref_orth * gd_q1
                + pref_parallel * gp_q1;

            gradients[j + 1] +=
                pref_orth * gd_q2
                + pref_parallel * gp_q2;
        }
    }
}

int main()
{
    int n_cells = 201;
    real l0 = 0.2;
    real K = 1.0;
    real A0 = 1.0;
    real gamma_a0 = 10.0;
    real gamma_b0 = gamma_a0;
    real h0 = 16.0;
    real w0 = 1.0;
    real K_bd = 100.0;
    int num_gauss_legendre_points = 32;
    lvm model(n_cells, l0, K, h0, w0, gamma_a0, gamma_b0, K_bd, num_gauss_legendre_points);
    printf("Initialization done\n");
    real dt = 1e-5;

    real adhesion_strength = 1.0;
    real adhesion_r0 = 0.5;
    real adhesion_inv_width = 1/1.0;
    real adhesion_cutoff = 5.0*adhesion_r0;

    int num_steps = int(5000/dt);

    int central_cell = n_cells/2;
    int i0 = central_cell - 4;
    int i1 = central_cell + 4 + 1;
    //model.scale_gamma(4.0, i0, i1, true);
    model.scale_gamma(0.25, i0, i1, false);
    model.scale_gamma_l(4.0, i0, i1+1);

    std::ofstream file("dump/output.bin", std::ios::binary);
    for(int step = 0; step < num_steps; step++)
    {
        if(step % 200000 == 0)
        {
            file.write(reinterpret_cast<const char*>(model.apical), sizeof(vec2) * (n_cells + 1));
            file.write(reinterpret_cast<const char*>(model.basal), sizeof(vec2) * (n_cells + 1));
            file.flush();
            printf("Time" ": %f\n", step*dt);
        }
        
        
        model.tension_step();
        model.area_step();
        model.boundary_step(-3.0, h0, false);
        segment_to_segment_geom_interaction(adhesion_strength, 
            adhesion_r0, 
            adhesion_r0, 
            adhesion_cutoff, 
            model.apical, 
            n_cells, 
            model.apical_interaction_grads);
        model.step(dt, true);
    };
    
    file.close();
}