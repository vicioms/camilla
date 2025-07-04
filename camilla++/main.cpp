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

int main()
{
    int num_quads = 5;
    vector<vector<int>> quads_conn;
    vector<float3> vertices;
    create_quad_chain(num_quads, 1.0, 16.0, quads_conn, vertices);
    vector<polygon*> polygons;
    vector<polygon*> boundaries;
    load_polygons(quads_conn, polygons, boundaries);
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