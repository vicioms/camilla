#include "../generic.h"

void get_curvature(const float2 r, const float2 r_prev, const float2 r_next,
                   bool is_left_boundary, bool is_right_boundary,
                   float& curvature, float2& c_grad, float2& c_grad_prev, float2& c_grad_next)
{
    float2 dr_centered = make_float2(r_next.x - r_prev.x, r_next.y - r_prev.y);
    float2 d2r = make_float2(r_next.x - 2*r.x + r_prev.x, r_next.y - 2*r.y + r_prev.y);
    float denom = powf(dr_centered.x*dr_centered.x + dr_centered.y*dr_centered.y, 1.5f);
    curvature = 4*(dr_centered.x*d2r.y - dr_centered.y*d2r.x) / denom;
    c_grad_prev = make_float2(0.0f, 0.0f);
    c_grad_next = make_float2(0.0f, 0.0f);
    // the denominator of the curvature is a function of r_prev, r_next
    // the numerator is a function of r, r_prev, r_next
    // so the gradient wrt r is just the numerator divided by the denominator
    c_grad = make_float2(dr_centered.y, -dr_centered.x) * (8.0f/denom);
    if(!is_left_boundary)
    {
        c_grad_prev = 
    };

};


void compute_curvatures(float2* r, float* curvatures, float2* curvatures_grads, int n, const float2 left_boundary, const float2 right_boundary)
{
    for(int i = 0; i < n; i++)
    {
        float2 r_prev = (i == 0) ? left_boundary : r[i-1];
        float2 r_next = (i == n-1) ? right_boundary : r[i+1];
        float2 dr_centered = make_float2(r_next.x - r_prev.x, r_next.y - r_prev.y);
        float2 d2r = make_float2(r_next.x - 2*r[i].x + r_prev.x, r_next.y - 2*r[i].y + r_prev.y);
        float denom = dr_centered.x*dr_centered.x + dr_centered.y*dr_centered.y;
        curvatures[i] = 4*(dr_centered.x*d2r.y - dr_centered.y*d2r.x) / powf(denom, 1.5f);
        curvatures_grads[i] = make_float2(0.0f, 0.0f);

    };
};

int main()
{
    float l0 = 1.0f;
    int num_vertices = 100;
    float2* positions;
    float* curvatures;
    positions = (float2*)malloc(sizeof(float2)*num_vertices);
    curvatures = (float*)malloc(sizeof(float)*num_vertices);
    for(int i = 0; i < num_vertices; i++)
    {
        positions[i] = make_float2(i*l0, 0);
        curvatures[i] = 0.0f;
    };
}