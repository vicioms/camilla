#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <tuple>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
using namespace std;


void trimesh_host_to_device(const trimesh& h_mesh, trimesh& d_mesh)
{
    // Allocate and copy vertices
    cudaMalloc(&d_mesh.vertices, sizeof(float3) * h_mesh.num_vertices);
    cudaMemcpy(d_mesh.vertices, h_mesh.vertices, sizeof(float3) * h_mesh.num_vertices, cudaMemcpyHostToDevice);

    // Allocate and copy triangles
    cudaMalloc(&d_mesh.triangles, sizeof(int3) * h_mesh.num_triangles);
    cudaMemcpy(d_mesh.triangles, h_mesh.triangles, sizeof(int3) * h_mesh.num_triangles, cudaMemcpyHostToDevice);

    // Edge arrays
    cudaMalloc(&d_mesh.edge_from, sizeof(int) * h_mesh.num_edges);
    cudaMemcpy(d_mesh.edge_from, h_mesh.edge_from, sizeof(int) * h_mesh.num_edges, cudaMemcpyHostToDevice);

    cudaMalloc(&d_mesh.edge_to, sizeof(int) * h_mesh.num_edges);
    cudaMemcpy(d_mesh.edge_to, h_mesh.edge_to, sizeof(int) *h_mesh.num_edges, cudaMemcpyHostToDevice);

    cudaMalloc(&d_mesh.edge_other, sizeof(int) * h_mesh.num_edges);
    cudaMemcpy(d_mesh.edge_other, h_mesh.edge_other, sizeof(int) * h_mesh.num_edges, cudaMemcpyHostToDevice);

    cudaMalloc(&d_mesh.edge_triangle, sizeof(int) * h_mesh.num_edges);
    cudaMemcpy(d_mesh.edge_triangle, h_mesh.edge_triangle, sizeof(int) * h_mesh.num_edges, cudaMemcpyHostToDevice);

    // Meta
    d_mesh.num_vertices = h_mesh.num_vertices;
    d_mesh.num_triangles = h_mesh.num_triangles;
    d_mesh.num_edges = h_mesh.num_edges;
};
void trimesh_prepare(float3* verts, int3* tris, 
    int num_vertices, int num_triangles, 
    trimesh& out_mesh,
    vector<vector<int>>& boundary_loops_vertices,
    vector<vector<int>>& boundary_loops_edge_indices,
    bool copyData)
{
    if(copyData) {
        out_mesh.vertices = (float3*)malloc(sizeof(float3) * num_vertices);
        out_mesh.triangles = (int3*)malloc(sizeof(int3) * num_triangles);
        memcpy(out_mesh.vertices, verts, sizeof(float3) * num_vertices);
        memcpy(out_mesh.triangles, tris, sizeof(int3) * num_triangles);
    } else {
        out_mesh.vertices = verts;
        out_mesh.triangles = tris;
    }

    out_mesh.num_vertices = num_vertices;
    out_mesh.num_triangles = num_triangles;

    std::vector<std::tuple<int, int, int, int>> edge_list;
    for (int i = 0; i < num_vertices; i++) 
    {
        edge_list.emplace_back(i,i,-1,-1);
    }
    unordered_map<int, int> unique_edges_counts;
    for (int t = 0; t < num_triangles; t++) {
        int3 tri = out_mesh.triangles[t];

        edge_list.emplace_back(tri.x, tri.y, tri.z, t);
        edge_list.emplace_back(tri.y, tri.z, tri.x, t);
        edge_list.emplace_back(tri.z, tri.x, tri.y, t);
        unique_edges_counts[min(tri.x,tri.y)*num_vertices + max(tri.x,tri.y)]++;
        unique_edges_counts[min(tri.y,tri.z)*num_vertices + max(tri.y,tri.z)]++;
        unique_edges_counts[min(tri.z,tri.x)*num_vertices + max(tri.z,tri.x)]++;
    };

    std::sort(edge_list.begin(), edge_list.end(),
        [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) < std::get<0>(b);
            return std::get<1>(a) < std::get<1>(b);
        });

    out_mesh.num_edges = edge_list.size();

    out_mesh.edge_from = (int*)malloc(sizeof(int) * out_mesh.num_edges);
    out_mesh.edge_to = (int*)malloc(sizeof(int) * out_mesh.num_edges);
    out_mesh.edge_other = (int*)malloc(sizeof(int) * out_mesh.num_edges);
    out_mesh.edge_triangle = (int*)malloc(sizeof(int) * out_mesh.num_edges);

    vector<pair<int2, int>> boundary_vertices_edges;
    for (int e = 0; e < out_mesh.num_edges; ++e) {
        int from  = std::get<0>(edge_list[e]);
        int to    = std::get<1>(edge_list[e]);
        int other = std::get<2>(edge_list[e]);
        int tri   = std::get<3>(edge_list[e]);

        if(from != to)
        {
            int num_unique_edges = unique_edges_counts[min(from,to)*num_vertices + max(from,to)];
            if(num_unique_edges == 1)
            {
                int2 packed_edge = {from, to};
                boundary_vertices_edges.emplace_back(packed_edge, e);
            };
        };

        out_mesh.edge_from[e] = from;
        out_mesh.edge_to[e] = to;
        out_mesh.edge_other[e] = other;
        out_mesh.edge_triangle[e] = tri;
    };
    vector<vector<pair<int2,int>>> boundary_loops;
    vector<pair<int2,int>> current_loop;
    pair<int2, int> current_edge;
    while(boundary_vertices_edges.size()>0)
    {
        if(current_loop.size() == 0)
        {
            current_edge = boundary_vertices_edges.front();
            boundary_vertices_edges.erase(boundary_vertices_edges.begin());
            current_loop.push_back(current_edge);
            //cout << current_edge.first.x << " " << current_edge.first.y << endl;
        }
        else
        {
            bool found_something = false;
            auto it = boundary_vertices_edges.begin();
            while (it != boundary_vertices_edges.end()) {
                if (it->first.x == current_edge.first.y) {
                    current_loop.push_back(*it);
                    current_edge = *it; // update current_edge to continue the chain
                    //cout << current_edge.first.x << " " << current_edge.first.y << endl;
                    it = boundary_vertices_edges.erase(it); // erase and get new iterator
                    found_something = true;
                    break;
                } else {
                    ++it;
                }
            };
            if(found_something == false)
            {
                boundary_loops.push_back(current_loop);
                current_loop.clear();
            };
        };
    };
    if(current_loop.size()>0)
    {
        boundary_loops.push_back(current_loop);
    };
    boundary_loops_vertices.clear();
    boundary_loops_edge_indices.clear();
    for(vector<pair<int2,int>> loop : boundary_loops)
    {
        vector<int> loop_vertices;
        vector<int> loop_edge_indices;
        for(pair<int2,int> edge : loop)
        {
            loop_vertices.push_back(edge.first.x);
            loop_edge_indices.push_back(edge.second);
        };
        boundary_loops_vertices.push_back(loop_vertices);
        boundary_loops_edge_indices.push_back(loop_edge_indices);
    };
};