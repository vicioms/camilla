#include "halfedge.h"
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}
void read_basic_ply(string filename, vector<float3>& vertices, vector<int3>& triangles)
{
    ifstream ifs;
    ifs.open(filename);
    std::string line;
    map<string, int> elements;
    while(std::getline(ifs, line))
    {
        if(line == "ply")
        {
            continue;
        }
        else if(line == "end_header")
        {
            break;
        }
        else
        {
            if(line.find("element", 0) == 0)
            {
                vector<string> bits = split(line, " ");
                if(bits.size() == 3)
                {
                    elements[bits[1]] = stoi(bits[2]);
                }
            }
        }
    };
    int num_vertices = elements["vertex"];
    int num_triangles = elements["face"];
    vertices.reserve(num_vertices);
    triangles.reserve(num_triangles);
    int v_idx = 0;
    int t_idx = 0;
    while(std::getline(ifs, line))
    {
        vector<string> bits = split(line, " ");
        if(bits.size() == 3)
        {   
            vertices.push_back({stof(bits[0]), stof(bits[1]), stof(bits[2])});
            v_idx += 1;
        }
        else if(bits.size() == 4)
        {
            if(bits[0] == "3")
            {
                triangles.push_back({stoi(bits[1]), stoi(bits[2]), stoi(bits[3])});
                t_idx += 1;
            }
        }
        else
        {
            continue;
        }
    }
    ifs.close();
} 

const int N_gl = 5;

const float s_gl_01[N_gl] = {
    0.046910077030668,
    0.230765344947158,
    0.5,
    0.769234655052841,
    0.953089922969332
};

const double w_gl_01[N_gl] = {
    0.118463442528095,
    0.239314335249683,
    0.284444444444444,
    0.239314335249683,
    0.118463442528095
};


float adhesion_helper_func(float d2, float d02)
{
    float ratio_squares = (d02/d2);
    float ratio_squares_squared = ratio_squares*ratio_squares;
    return -(2/d2)*ratio_squares_squared*(2*ratio_squares_squared - 1.0f);
};

int main()
{
    float x0 = -0.1;
    float y0 = 0.1;
    float dx = 0.5;
    float dy = 1.0;
    float L = 2.0;

    float d0 = 0.2;

    float dt = 0.01;

    float tension = 0.01;

    ofstream file;
    file.open("test.txt");
    for(int i = 0; i < 100000; i++)
    {
        file << x0 << " " << y0 << " " << dx << " " << dy << " " << L << endl;
        float F_x0 = 0.0;
        float F_y0 = 0.0;
        float F_dx = 0.0;
        float F_dy = 0.0;
        float F_L = 0.0;
        for(int k_1 = 0; k_1 < N_gl; k_1++)
        {
            for(int k_2 = 0; k_2 < N_gl; k_2++)
            {
                float delta_x = (x0 + dx*s_gl_01[k_1] - s_gl_01[k_2]*L);
                float delta_y = y0 + dy*s_gl_01[k_1];
                float d2 =  delta_x*delta_x + delta_y*delta_y;
                float integrand = -adhesion_helper_func(d2, d0*d0)*w_gl_01[k_1]*w_gl_01[k_2];
                F_x0 += delta_x*integrand;
                F_y0 += delta_y*integrand;
                F_dx += s_gl_01[k_1]*delta_x*integrand;
                F_dy += s_gl_01[k_1]*delta_y*integrand;
                F_L  += (-s_gl_01[k_2])*(delta_x*integrand);
            }
        };

        x0 += dt*F_x0;
        y0 += dt*F_y0;
        dx += dt*(F_dx - tension*dx/(sqrtf(dx*dx+dy*dy)));
        dy += dt*(F_dy - tension*dy/(sqrtf(dx*dx+dy*dy)));
        L += dt*(F_L - tension);
    };
    file.close();



    //int num_quads = 5;
    //vector<vector<int>> quads_conn;
    //vector<float3> vertices;
    //create_quad_chain(num_quads, 1.0, 16.0, quads_conn, vertices);
    //vector<polygon*> polygons;
    //vector<polygon*> boundaries;
    //load_polygons(quads_conn, polygons, boundaries);
    return 0;
};

int custom_main()
{
    vector<float3> vertices;
    vector<float3> gradients;
    vertices.emplace_back(0.0f, 0.0f,0.0f);
    vertices.emplace_back(1.0f, 0.0f,0.0f);
    vertices.emplace_back(1.0f, 1.0f,0.0f);
    vertices.emplace_back(0.0f, 1.0f,0.0f);
    for(int i = 0; i < vertices.size(); i++)
    {
        gradients.emplace_back(0.0f,0.0f,0.0f);
    };
    vector<vector<int>> cty;
    cty.push_back({0,1,2,3});
    vector<polygon*> polygons;
    vector<polygon*> boundaries;
    load_polygons(cty, polygons, boundaries);
    polygons[0]->area_grad(vertices, gradients);
    for(float3 grad : gradients)
    {
        cout << grad.x << " " << grad.y << " " << grad.z << endl; 
    };
    return 0;
};

int tri_main()
{
    vector<float3> vertices;
    vector<int3> triangles;
    read_basic_ply("test.ply", vertices, triangles);
    vector<vector<int>> polygons_connectivity;
    for(auto tri : triangles)
    {
        vector<int> tri_v = {tri.x, tri.y, tri.z};
        polygons_connectivity.push_back(tri_v);
    }
   
    //vector<vector<int>> polygons_connectivity;
   //polygons_connectivity.push_back({0,1,2,3,4,5});
   //polygons_connectivity.push_back({2,6,7,8,3});
   //polygons_connectivity.push_back({3,8,9,4});
   vector<polygon*> polygons;
   vector<polygon*> boundaries;
   load_polygons(polygons_connectivity, polygons, boundaries);
   for(auto ptr : boundaries)
   {
        float3 area_vector = ptr->area_vector(vertices);
        cout << area_vector.x << " " << area_vector.y << " " << area_vector.z << endl;
        //ptr->apply_edges([](halfedge* e){
        //    cout << e->source << " " << e->target << endl;});   
        cout << "---------------" << endl;
   };
   //for(auto ptr : polygons)
   //{
   //     ptr->apply_edges([](halfedge* e){
   //         halfedge_apply_neighbourhood(e, 
   //         [](halfedge* e) { cout << e->source << " " << e->target << endl;});
   //         cout << "...." << endl;
   //     });        
   //     cout << "---------------" << endl;
   //}
   return 0;
};