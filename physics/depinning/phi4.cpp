#include<vector>
#include<functional>
#include<unordered_map>
#include<fstream>
#include<random>
#include<limits>
#include<set>
#include<string>
#include<iostream>
#include<sstream>
#include "../generic.h"
#include "../argparser.h"

using namespace std;
static constexpr float inf = std::numeric_limits<float>::infinity();

void initialize_bubble(float* phi, int L, int L0)
{
    const float cx = (L - 1) / 2.0f;
    const float cy = (L - 1) / 2.0f;

    const float R = (L & 1) ? (L0 + 1.0f) : (L0 + 2.5f);
    const float R2 = R * R;

    for (int i = 0; i < L; ++i)
    {
        const float dx = i - cx;
        for (int j = 0; j < L; ++j)
        {
            const float dy = j - cy;
            const float dist2 = dx*dx + dy*dy;

            const int idx = pack(i, j, L);
            phi[idx] = (dist2 <= R2) ? 1.0f : -1.0f;
        }
    }
};

void initialize_disorder(float* r, int L, float delta)
{
    default_random_engine eng;
    normal_distribution<float> distr;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            int idx = pack(i, j, L);
            //r[idx] = distr(eng)*2*delta - delta;
            r[idx]=distr(eng)*delta;
        }
    }
}

static inline float cbrt_signedf(float x) {
    return copysignf(cbrtf(fabsf(x)), x);
}

static inline int solve_depressed_cubicf(float p, float q, float r[3]) {
    // Discriminant: Δ = (q/2)^2 + (p/3)^3
    float half_q = 0.5f * q;
    float third_p = p / 3.0f;
    float disc = half_q*half_q + third_p*third_p*third_p;

    if (disc > 0.0f) {
        // One real root
        float s = sqrtf(disc);
        float u = cbrt_signedf(-half_q + s);
        float v = cbrt_signedf(-half_q - s);
        r[0] = u + v;
        return 1;
    } else if (disc == 0.0f) {
        // Multiple real roots, at least two equal
        float u = cbrt_signedf(-half_q);
        r[0] = 2.0f*u;
        r[1] = -u;
        return 2; // two distinct values (one is double root)
    } else {
        // Three distinct real roots (casus irreducibilis)
        float rho = 2.0f * sqrtf(-third_p);
        float acos_arg = (-half_q) / sqrtf(-third_p*third_p*third_p); // = (3q)/(2p)*sqrt(-3/p)
        // Clamp for safety
        if (acos_arg > 1.0f)  acos_arg = 1.0f;
        if (acos_arg < -1.0f) acos_arg = -1.0f;
        float theta = acosf(acos_arg);
        r[0] = rho * cosf( theta/3.0f);
        r[1] = rho * cosf((theta + 2.0f*M_PI)/3.0f);
        r[2] = rho * cosf((theta + 4.0f*M_PI)/3.0f);
        return 3;
    }
}


float solve(float* phi, float* solutions, int L, float c, float e0, float* disorder, float h, float noise)
{
    default_random_engine eng;
    normal_distribution<float> distr;
    float op = 0.0;
    float neighbors_sum;
    for(int idx = 0; idx < L*L; idx++)
    {
        solutions[idx] = 0.0;
        int r,c;
        unpack(idx, L, r, c);
        neighbors_sum = 0.0;
        neighbors_sum += r == 0 ? -1 : phi[pack(r-1, c, L)];
        neighbors_sum += r == L-1 ? -1 : phi[pack(r+1, c, L)];
        neighbors_sum += c == 0 ? -1 : phi[pack(r, c-1, L)];
        neighbors_sum += c == L-1 ? -1 : phi[pack(r, c+1, L)];
        //-e0 phi^3 + (e0(1+r)-4) phi + phin + h = 0
        // phi^3 + (4/e0 - (1+r)) phi - (phin+h)/e0 = 0 
        float p = (4/e0) -(1.0f + disorder[idx]);
        float q = -(h+neighbors_sum + distr(eng)*noise)/e0;
        float phi_sols[3];
        int num_solutions = solve_depressed_cubicf(p, q, phi_sols);
        float best_solution_distance = inf;
        int best_solution_idx = -1;
        for(int s = 0; s < num_solutions; s++)
        {
            float sol_distance = fabsf(phi[idx] - phi_sols[s]);
            if(sol_distance < best_solution_distance)
            {
                best_solution_distance = sol_distance;
                best_solution_idx = s;
            }
        };
        solutions[idx] = phi_sols[best_solution_idx];
    };

     for (int idx = 0; idx < L*L; ++idx) {
        phi[idx] = solutions[idx];
        op += phi[idx];
    };
    return op/(L*L);
}; 

int main(int argc, char** argv)
{
    ArgParser parser(argc,argv);
    float g;
    float c;
    float e0;
    float h;
    float delta;
    int L;
    int L0;
    L = parser.get_int("L", 1024);
    L0 = parser.get_float("L0", 64);
    g = parser.get_float("g", 1.0f);
    c = parser.get_float("c", 1.0f);
    e0 = parser.get_float("e0", 1.0f);
    h = parser.get_float("h", 0.15f);
    delta = parser.get_float("delta", 0.5f);
    

    float* phi = (float*)malloc(sizeof(float)*L*L);
    float* cubic_solutions = (float*)malloc(sizeof(float)*L*L);
    initialize_bubble(phi, L, L0);
    float* r = (float*)malloc(sizeof(float)*L*L);
    initialize_disorder(r, L, delta);
    std::ofstream out("phi4.bin", std::ios::binary);
    float old_op = inf;
    for(int time_step = 0; time_step < 10000; time_step++)
    {
        if(time_step % 10 == 0)
        {
            cout << time_step << endl;
            out.write((char*)phi, sizeof(float) * L * L);
            out.flush();
        }
        
        float new_op = solve(phi, cubic_solutions, L, c, e0, r, h, 0.1);
        //if(fabsf(new_op-old_op) < 1e-3)
        //{
        //    
        //    h *= 1.01;
        //    cout << "New field: " << h << endl;
        //}
        old_op = new_op;
    };
    out.close();
}