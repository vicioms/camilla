#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
using namespace std;

/*
double adhesion_potential_gradients(double a, double x0, double y0, double dx, double dy, double L, double* gradients)
{
    double b1 = erf((L-x0)/(sqrt(2)*a)) + erf((dx-x0)/(sqrt(2)*a));
    double b2 = erf(y0/(sqrt(2)*a)) - erf((dy-y0)/(sqrt(2)*a));
    double b3 = -a*a*M_PI/(2*dy*L);
    double d_b12_d_L = b2*exp(-(L-x0)*(L-x0)/(2*a*a))*sqrt(2/M_PI)/a;
    double d_b12_d_x0 = -b2*(sqrt(2/M_PI)/a)*(exp(-(L-x0)*(L-x0)/(2*a*a)) + exp(-(dx-x0)*(dx-x0)/(2*a*a)));
    double d_b12_d_dx = b2*(sqrt(2/M_PI)/a)*exp(-(dx-x0)*(dx-x0)/(2*a*a));
    double d_b12_d_y0 = b1*(sqrt(2/M_PI)/a)*(exp(-y0*y0/(2*a*a)) + exp(-(dy-y0)*(dy-y0)/(2*a*a)));
    double d_b12_d_dy = -b1*(sqrt(2/M_PI)/a)*exp(-(dy-y0)*(dy-y0)/(2*a*a));
    double d_b3_d_L = a*a*M_PI/(2*dy*L*L);
    double d_b3_d_dy = a*a*M_PI/(2*dy*dy*L);
    gradients[0] = b3*d_b12_d_x0; //x0
    gradients[1] = b3*d_b12_d_y0; //y0
    gradients[2] = b3*d_b12_d_dx; //dx
    gradients[3] = b3*d_b12_d_dy + d_b3_d_dy*b1*b2; //dy
    gradients[4] = b3*d_b12_d_L + d_b3_d_L*b1*b2; //L;
    return b1*b2*b3;
};

void adhesion_pipeline(double a, double x1_1, double y1_1, double x2_1, double y2_1,
                       double x1_2, double y1_2, double x2_2, double y2_2, double* gradients)
{

    double dx1 = x2_1 - x1_1;
    double dy1 = y2_1 - y1_1;
    double dx2 = x2_2 - x1_2;
    double dy2 = y2_2 - y1_2;
    
    double L = sqrt(dx1*dx1+dy1*dy1);
    double cos_theta = dx1/L;
    double sin_theta = -dy1/L;
    double x0 = x1_2 - x1_1;
    double y0 = y1_2 - y1_1;
    double dx = (dx1*dx2 + dy1*dy2)/L; //cos_theta*dx2 - sin_theta*dy2;
    double dy = (-dy1*dx2 + dx1*dy2)/L;  //sin_theta*dx2 + cos_theta*dy2;

    double* transformed_grads = new double[5];
    adhesion_potential_gradients(a, x0,y0, dx, dy, L, transformed_grads);
    
    double d_x0_d_x1_2 = 1.0;
    double d_x0_d_x1_1 = -1.0;
    double d_y0_d_y1_2 = 1.0;
    double d_y0_d_y1_1 = -1.0;

    double d_L_dx1 = dx1/L;
    double d_L_dy1 = dy1/L;

    double d_L_d_x2_1 = d_L_dx1;
    double d_L_d_x1_1 = -d_L_dx1;
    double d_L_d_y2_1 = d_L_dy1;
    double d_L_d_y1_1 = -d_L_dy1;

    double d_dx_d_dx1 = 0.0; 
    double d_dx_d_dy1 = 0.0; 
    double d_dx_d_dx2 = 0.0; 
    double d_dx_d_dy2 = 0.0; 
    double d_dy_d_dx1 = 0.0; 
    double d_dy_d_dy1 = 0.0; 
    double d_dy_d_dx2 = 0.0; 
    double d_dy_d_dy2 = 0.0; 
    delete[] transformed_grads;
};*/

// Function to compute adhesion potential gradients
double adhesion_potential_gradients(double a, double x0, double y0, double dx, double dy, double L, double* gradients)
{
    double b1 = erf((L-x0)/(sqrt(2)*a)) + erf((dx-x0)/(sqrt(2)*a));
    double b2 = erf(y0/(sqrt(2)*a)) - erf((dy-y0)/(sqrt(2)*a));
    double b3 = -a*a*M_PI/(2*dy*L);
    double d_b12_d_L = b2*exp(-(L-x0)*(L-x0)/(2*a*a))*sqrt(2/M_PI)/a;
    double d_b12_d_x0 = -b2*(sqrt(2/M_PI)/a)*(exp(-(L-x0)*(L-x0)/(2*a*a)) + exp(-(dx-x0)*(dx-x0)/(2*a*a)));
    double d_b12_d_dx = b2*(sqrt(2/M_PI)/a)*exp(-(dx-x0)*(dx-x0)/(2*a*a));
    double d_b12_d_y0 = b1*(sqrt(2/M_PI)/a)*(exp(-y0*y0/(2*a*a)) + exp(-(dy-y0)*(dy-y0)/(2*a*a)));
    double d_b12_d_dy = -b1*(sqrt(2/M_PI)/a)*exp(-(dy-y0)*(dy-y0)/(2*a*a));
    double d_b3_d_L = a*a*M_PI/(2*dy*L*L);
    double d_b3_d_dy = a*a*M_PI/(2*dy*dy*L);
    gradients[0] = b3*d_b12_d_x0; // x0
    gradients[1] = b3*d_b12_d_y0; // y0
    gradients[2] = b3*d_b12_d_dx; // dx
    gradients[3] = b3*d_b12_d_dy + d_b3_d_dy*b1*b2; // dy
    gradients[4] = b3*d_b12_d_L + d_b3_d_L*b1*b2; // L
    return b1*b2*b3;
};

double simplified_adhesion_potential_gradients(double a, bool decreasing_function, float power,  double x0, double y0, double dx, double dy, double L, double* gradients)
{
    double argument = 0.0;
   
    if((decreasing_function and dy<0) or (!decreasing_function and dy > 0))
    {
        argument = sqrt((x0+dx)*(x0+dx)+(y0+dy)*(y0+dy));
    }
    else
    {
        argument = sqrt(x0*x0+y0*y0);
    };
    double function = 1/pow(argument, power);
    double d_function =  -(power/argument)*function;
    if(!decreasing_function)
    {
        function *= -1.0;
        d_function *= -1.0;
    };
    if((decreasing_function and dy<0) or (!decreasing_function and dy > 0))
    {
        
    }
    else
    {

    };

};

// Adhesion pipeline function
void adhesion_pipeline(double a, 
                       double x1_1, double y1_1, double x2_1, double y2_1,
                       double x1_2, double y1_2, double x2_2, double y2_2, double* gradients)
{
    double dx1 = x2_1 - x1_1;
    double dy1 = y2_1 - y1_1;
    double dx2 = x2_2 - x1_2;
    double dy2 = y2_2 - y1_2;
    
    double L = sqrt(dx1 * dx1 + dy1 * dy1);
    double cos_theta = dx1 / L;
    double sin_theta = -dy1 / L;
    
    double x0 = x1_2 - x1_1;
    double y0 = y1_2 - y1_1;
    
    double dx = (dx1 * dx2 + dy1 * dy2) / L;
    double dy = (-dy1 * dx2 + dx1 * dy2) / L;
    
    double transformed_grads[5];
    adhesion_potential_gradients(a, x0, y0, dx, dy, L, transformed_grads);
    double d_E_d_x0 = transformed_grads[0];
    double d_E_d_y0 = transformed_grads[1];
    double d_E_d_dx = transformed_grads[2];
    double d_E_d_dy = transformed_grads[3];
    double d_E_d_L = transformed_grads[4];

    double* L_grads = new double[8];
    L_grads[0] = -dx1/L;
    L_grads[1] = -dy1/L;
    L_grads[2] = dx1/L;
    L_grads[3] = dy1/L;
    L_grads[4] = 0.0;
    L_grads[5] = 0.0;
    L_grads[6] = 0.0;
    L_grads[7] = 0.0;
    double* x0_grads = new double[8];
    x0_grads[0] = -1.0;
    x0_grads[1] = 0.0;
    x0_grads[2] = 0.0;
    x0_grads[3] = 0.0;
    x0_grads[4] = 1.0;
    x0_grads[5] = 0.0;
    x0_grads[6] = 0.0;
    x0_grads[7] = 0.0;
    double* y0_grads = new double[8];
    y0_grads[0] = 0.0;
    y0_grads[1] = -1.0;
    y0_grads[2] = 0.0;
    y0_grads[3] = 0.0;
    y0_grads[4] = 0.0;
    y0_grads[5] = 1.0;
    y0_grads[6] = 0.0;
    y0_grads[7] = 0.0;

    double* dx_grads = new double[8];

    double L2 = L * L;
    
    dx_grads[0] = -dx2/L + dx*dx1/L2;
    dx_grads[1] = -dy2/L + dx*dy1/L2;
    dx_grads[2] = dx2/L - dx*dx1/L2;
    dx_grads[3] = dy2/L - dx*dy1/L2;
    dx_grads[4] = -dx1/L;
    dx_grads[5] = -dy1/L;
    dx_grads[6] = dx1/L;
    dx_grads[7] = dy1/L;

    double* dy_grads = new double[8];
    
    dy_grads[0] = -dy2/L + dy*dx1/L2;
    dy_grads[1] = dx2/L + dy*dy1/L2;
    dy_grads[2] = dy2/L - dy*dx1/L2;
    dy_grads[3] = -dx2/L - dy*dy1/L2;
    dy_grads[4] = dy1/L;
    dy_grads[5] = -dx1/L;
    dy_grads[6] = -dy1/L;
    dy_grads[7] = dx1/L;

    for(int k = 0; k < 8; k++)
    {
        gradients[k] = d_E_d_x0*x0_grads[k]+ d_E_d_y0*y0_grads[k] + d_E_d_dx*dx_grads[k] +  d_E_d_dy*dy_grads[k] + d_E_d_L*L_grads[k] ;
    };
};


void init(double* x_a, double* x_b, double* y_a, double* y_b, double w, double h, int N)
{
    for(int i = 0; i < N; i++)
    {
        x_a[i] = i*w;
        x_b[i] = i*w;
        y_a[i] = h;
        y_b[i] = 0.0f;
    }
};
void set(double* x, double v, int N)
{
    for(int i = 0; i < N; i++)
    {
        x[i] = v;  
    }
};

void set(double* x, double v, int i_start, int i_end)
{
    for(int i = i_start; i < i_end; i++)
    {
        x[i] = v;  
    }
};
void zero(double* x, int N)
{
    for(int i = 0; i < N; i++)
    {
        x[i] = 0.0f;
    };
}
void compute_areas(double* x_a, double* x_b, double* y_a, double* y_b, double* areas, int N)
{
    for(int i = 0; i < N-1; i++)
    {
        areas[i] =  x_b[i]*y_b[i+1] - x_b[i+1]*y_b[i];
        areas[i] += x_b[i+1]*y_a[i+1] - x_a[i+1]*y_b[i+1];
        areas[i] += x_a[i+1]*y_a[i] - x_a[i]*y_a[i+1];
        areas[i] += x_a[i]*y_b[i] - x_b[i]*y_a[i];
        areas[i] *= 0.5f;
    }
};
void compute_lengths(double* x_a, double* x_b, double* y_a, double* y_b, double* l_a, double* l_b, double* l_l, int N)
{
    for(int i = 0; i < N; i++)
    {
        l_l[i] = y_a[i]-y_b[i];
        if(i < N-1)
        {
            l_a[i] = x_a[i+1]-x_a[i];
            l_b[i] = x_b[i+1]-x_b[i];
        }
    }
};
void accumulate_area_gradients(double* x_a, double* x_b, double* y_a, double* y_b, 
                               double* dx_a, double* dx_b, double* dy_a, double* dy_b, 
                               double* areas, double K, double A0, int N)
{
    for(int i = 0; i < N-1; i++)
    {
        double base_grad = K * (areas[i] - A0) * 0.5f;
        
        dx_a[i] += base_grad * (y_b[i] - y_a[i+1]);
        dx_a[i+1] += base_grad * (y_a[i] - y_b[i+1]);
        dx_b[i] += base_grad * (y_b[i+1] - y_a[i]);
        dx_b[i+1] += base_grad * (y_a[i+1] - y_b[i]);
        dy_a[i] += base_grad * (x_a[i+1] - x_b[i]);
        dy_a[i+1] += base_grad * (x_b[i+1] - x_a[i]);
        dy_b[i] += base_grad * (x_a[i] - x_b[i+1]);
        dy_b[i+1] += base_grad * (x_b[i] - x_a[i+1]);
    }
};
void accumulate_lateral_tension_gradients(double* x_a, double* x_b, double* y_a, double* y_b, double* dx_a, double* dx_b, double* dy_a, double* dy_b, double* r_l, double l0, int N)
{
    for(int i = 0; i < N; i++)
    {
        double l_i = sqrt((y_a[i] - y_b[i])*(y_a[i] - y_b[i]) +  (x_a[i] - x_b[i])*(x_a[i] - x_b[i]));
        if(l_i > l0)
        {
            dx_a[i] += r_l[i]* (x_a[i] - x_b[i])/l_i;
            dx_b[i] += r_l[i]* (x_b[i] - x_a[i])/l_i;
            dy_a[i] += r_l[i]* (y_a[i] - y_b[i])/l_i;
            dy_b[i] += r_l[i]* (y_b[i] - y_a[i])/l_i;
        }
        else
        {
            dx_a[i] += r_l[i]* (x_a[i] - x_b[i])/l0;
            dx_b[i] += r_l[i]* (x_b[i] - x_a[i])/l0;
            dy_a[i] += r_l[i]* (y_a[i] - y_b[i])/l0;
            dy_b[i] += r_l[i]* (y_b[i] - y_a[i])/l0;
        }
    };
};
void accumulate_horizontal_tension_gradients(double* x, double* y, double* dx, double* dy, double* r, double l0, int N)
{
    for(int i = 0; i < N-1; i++)
    {
        double l_i = sqrt((x[i+1] - x[i])*(x[i+1]-x[i]) +  (y[i+1] - y[i])*(y[i+1]-y[i]));
        if(l_i > l0)
        {
            dx[i]   += r[i]*(x[i] - x[i+1])/l_i;
            dx[i+1] += r[i]*(x[i+1] - x[i])/l_i;
            dy[i]   += r[i]*(y[i] - y[i+1])/l_i;
            dy[i+1] += r[i]*(y[i+1] - y[i])/l_i;
        }
        else
        {
            dx[i]   += (r[i]/l0)*(x[i] - x[i+1]);
            dx[i+1] += (r[i]/l0)*(x[i+1] - x[i]);
            dy[i]   += (r[i]/l0)*(y[i] - y[i+1]);
            dy[i+1] += (r[i]/l0)*(y[i+1] - y[i]);
        }
    };
};

void add_and_zero(double* x, double* y, double factor, int N)
{
    for(int i =0; i < N; i++)
    {
        x[i] += factor*y[i];
        y[i] = 0.0f;
    }
};

int main()
{
    //first segment
    double x1_1 = -1.0;
    double y1_1 = 0.0;
    double x2_1 = -3.0;
    double y2_1 = 2.0;
    //second segment
    double x1_2 = 1.0;
    double y1_2 = 0.0;
    double x2_2 = 3.0;
    double y2_2 = 2.0;
    double* xys = new double[8];
    xys[0] = x1_1;
    xys[1] = y1_1;
    xys[2] = x2_1;
    xys[3] = y2_1;
    xys[4] = x1_2;
    xys[5] = y1_2;
    xys[6] = x2_2;
    xys[7] = y2_2;
    std::ofstream outfile;
    outfile.open("hist.txt");

    double dt  = 1e-3;
    
    for(int n = 0; n < 1000000; n++)
    {
        double* grads_attr = new double[8];
        adhesion_pipeline(1.0, xys[0], xys[1], xys[2], xys[3], xys[4], xys[5], xys[6], xys[7],grads_attr);
        double* grads_rep = new double[8];
        adhesion_pipeline(0.5, xys[0], xys[1], xys[2], xys[3], xys[4], xys[5], xys[6], xys[7],grads_rep);
        for(int k = 0; k < 8; k++)
        {
            xys[k] -= dt*(grads_attr[k] - grads_rep[k]);
            outfile << xys[k] << " ";
        }
        outfile << endl;
        delete[] grads_attr;
        delete[] grads_rep;
    }
    outfile.close();
    return 0;
}   

int main2()
{
    int N = 101;
    int Nt = 4;
    double w = 1.0;
    double h = 16.0;
    double K = 1.0;
    double ra0 = 10.0;
    double rb0 = ra0;
    double rl0 = (2.0 * w * ra0 / h);
    double A0 = w * h + ra0 / K / h;
    double l0 = 0.2;
    double F = 1.0;
    double Ks = 100.0;

    double* x_a = new double[N];
    double* x_b = new double[N];
    double* y_a = new double[N];
    double* y_b = new double[N];
    double* dx_a = new double[N];
    double* dx_b = new double[N];
    double* dy_a = new double[N];
    double* dy_b = new double[N];

    double* r_a = new double[N-1];
    double* r_b = new double[N-1];
    double* r_l = new double[N];

    double x_L = 0.0f;
    double x_R = N*w;

    
    double* areas = new double[N-1];

    double dt = 1e-5;
    
    init(x_a, x_b, y_a, y_b, w, h, N);
    zero(dx_a,N); zero(dx_b,N); zero(dy_a,N); zero(dy_b,N);
    set(r_a, ra0, N-1);
    set(r_b, rb0, N-1);
    set(r_l, rl0, N);

    int center_vertex = (N-1)/2;
    set(r_b, ra0*0.4, center_vertex - Nt, center_vertex + Nt + 1);
    set(r_l, rl0*2.0, center_vertex - Nt, center_vertex + Nt);


    int steps = 40000000;
    int step_unfold = steps/3;

    int save_interval = 10000;

    std::ofstream outfile;
    outfile.open("vertices.txt");
    for(int step = 0; step < steps; step++)
    {
        if(step == step_unfold)
        {
            F = 0.0;
            set(r_b, rb0, N-1);
            set(r_l, rl0, N);
        }
        if(step % save_interval == 0)
        {
            cout << step << endl;
            for(int i = 0; i < N; i++)
            {
                outfile << x_a[i] << " " << y_a[i] << " " << x_b[i] << " " << y_b[i] << endl;        
            };
        }
        //zero(dx_a,N); zero(dx_b,N); zero(dy_a,N); zero(dy_b,N);
        compute_areas(x_a, x_b, y_a, y_b, areas, N);
        accumulate_area_gradients(x_a,x_b,y_a,y_b, dx_a, dx_b, dy_a, dy_b, areas, K, A0, N);
        accumulate_lateral_tension_gradients(x_a,x_b,y_a,y_b, dx_a, dx_b, dy_a, dy_b, r_l, l0, N);
        accumulate_horizontal_tension_gradients(x_a, y_a, dx_a, dy_a, r_a, l0, N);
        accumulate_horizontal_tension_gradients(x_b, y_b, dx_b, dy_b, r_b, l0, N);

        double dx_L = -F + Ks*(2*x_L-x_a[0]-x_b[0]);
        double dx_R = F + Ks*(2*x_R-x_a[N-1]-x_b[N-1]);
        dx_a[0] += Ks*(x_a[0]-x_L);
        dx_b[0] += Ks*(x_b[0]-x_L);
        dx_a[N-1] += Ks*(x_a[N-1]-x_R);
        dx_b[N-1] += Ks*(x_b[N-1]-x_R);
        x_L -= dt*dx_L;
        x_R -= dt*dx_R;

        add_and_zero(x_a, dx_a, -dt, N);
        add_and_zero(x_b, dx_b, -dt, N);
        add_and_zero(y_a, dy_a, -dt, N);
        add_and_zero(y_b, dy_b, -dt, N);
    };
    outfile.close();
    return 0;
};