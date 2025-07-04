#pragma once
#include "base.h"
#include <vector>
#include <functional>
#include <unordered_map>
#include <set>
#include <map>
#include <unordered_set>
#include <utility>
#include <string>
#include <iostream>
using namespace std;

namespace std {
    template <>
    struct hash<int2> {
        std::size_t operator()(const int2& k) const {
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
        }
    };
}

const int POLYGON_MAX_NUM_SIDES = 1000;

struct polygon;

struct halfedge {
    int source;
    int target;
    halfedge* prev;
    halfedge* next;
    halfedge* twin;
    polygon* parent;

    halfedge() : source(-1), target(-1), prev(nullptr), next(nullptr), twin(nullptr), parent(nullptr) {}

    halfedge(int source_, int target_)
        : source(source_), target(target_), prev(nullptr), next(nullptr), twin(nullptr), parent(nullptr) {}

    halfedge(int source_, int target_, halfedge* prev_, halfedge* next_)
        : source(source_), target(target_), prev(prev_), next(next_), twin(nullptr), parent(nullptr) {}

    halfedge(halfedge* prev_, halfedge* next_)
        : source(-1), target(-1), prev(prev_), next(next_), twin(nullptr), parent(nullptr) {}

    halfedge(int source_, int target_, halfedge* prev_, halfedge* next_, halfedge* twin_)
        : source(source_), target(target_), prev(prev_), next(next_), twin(twin_), parent(nullptr) {}

    halfedge(int source_, int target_, halfedge* prev_, halfedge* next_, halfedge* twin_, polygon* parent_)
        : source(source_), target(target_), prev(prev_), next(next_), twin(twin_), parent(parent_) {}

    void join_prev(halfedge* other) {
        if (other) {
            prev = other;
            other->next = this;
        }
    }

    void join_next(halfedge* other) {
        if (other) {
            next = other;
            other->prev = this;
        }
    }

    void join_twin(halfedge* other) {
        if (other) {
            twin = other;
            other->twin = this;
        }
    }

    size_t get_hash() const {
        return hash<int>()(source) ^ (hash<int>()(target) << 1);
    }

    size_t get_reverse_hash() const {
        return hash<int>()(target) ^ (hash<int>()(source) << 1);
    };

};

struct polygon {
    halfedge* root = nullptr;

    bool apply_edges(function<void(halfedge*)> func) {
        if (!root) return false;
        halfedge* current = root;
        int count = 0;
        do {
            if (++count > POLYGON_MAX_NUM_SIDES) return false;
            func(current);
            current = current->next;
        } while (current && current != root);
        return true;
    };

    bool apply_vertices(function<void(int)> func) {
        if (!root) return false;
        halfedge* current = root;
        int count = 0;
        do {
            if (++count > POLYGON_MAX_NUM_SIDES) return false;
            func(current->source);
            current = current->next;
        } while (current && current != root);
        return true;
    }

    template<typename acc_type>
    bool accumulate_edges(function<acc_type(halfedge*)> func, acc_type& value) const {
        if (!root) return false;
        halfedge* current = root;
        int count = 0;
        do {
            if (++count > POLYGON_MAX_NUM_SIDES) return false;
            value += func(current);
            current = current->next;
        } while (current && current != root);
        return true;
    };

    int size() const {
        int num_sides = 0;
        bool success = this->accumulate_edges<int>([](halfedge* e) {
            return 1;
        }, num_sides);
        return success ? num_sides : -1;
    }

    float3 area_vector(const vector<float3>& vertices) const {
        float3 result = zero3;
        bool success = this->accumulate_edges<float3>([&vertices](halfedge* e) {
            const float3& a = vertices[e->source];
            const float3& b = vertices[e->target];
            return cross(a, b);
        }, result);
        return success ? result * 0.5f : nan3;
    };
    bool area_grad(const vector<float3>& vertices,  vector<float3>& gradients)
    {
        float3 area_v = this->area_vector(vertices);
        if(area_v == nan3)
            return false;
        float3 normal_v = normalize(area_v);
        bool success = this->apply_edges([&vertices, &gradients, normal_v](halfedge* e) {
            int source = e->source;
            int target = e->target;
            float3 normal_cross_source = cross(normal_v, vertices[source]);
            float3 normal_cross_target = cross(normal_v, vertices[target]);
            gradients[source] += -0.5*normal_cross_target;
            gradients[target] += 0.5*normal_cross_source;
        });
        return success;
    };
    bool area_elasticity_grad(const vector<float3>& vertices,  vector<float3>& gradients, float k, float a0)
    {
        float3 area_v = this->area_vector(vertices);
        if(area_v == nan3)
            return false;
        float area = length(area_v);
        float3 normal_v = normalize(area_v);
        bool success = this->apply_edges([&vertices, &gradients, normal_v, area, k, a0](halfedge* e) {
            int source = e->source;
            int target = e->target;
            float3 normal_cross_source = cross(normal_v, vertices[source]);
            float3 normal_cross_target = cross(normal_v, vertices[target]);
            gradients[source] += (-0.5*k*(area-a0))*normal_cross_target;
            gradients[target] += (0.5*k*(area-a0))*normal_cross_source;
        });
        return success;
    };
    
    float perimeter(const vector<float3>& vertices) const {
        float result = 0.0f;
        bool success = this->accumulate_edges<float>([&vertices](halfedge* e) {
            return length(vertices[e->target] - vertices[e->source]);
        }, result);
        return success ? result : 0.0f;
    }
    bool perimeter_grad(const vector<float3>& vertices,  vector<float3>& gradients)
    {
        float perimeter = this->perimeter(vertices);
        if(perimeter < 0)
            return false;
        bool success = this->apply_edges([&vertices, &gradients, perimeter](halfedge* e) {
            int source = e->source;
            int target = e->target;
            
            const float3& a = vertices[source];
            const float3& b = vertices[target];

            float3 delta = (b-a)/perimeter;
            gradients[source] -= delta;
            gradients[target] += delta;
        });
        return success;
    };
    bool perimeter_tension_grad(const vector<float3>& vertices,  vector<float3>& gradients, float gamma)
    {
        float perimeter = this->perimeter(vertices);
        if(perimeter < 0)
            return false;
        bool success = this->apply_edges([&vertices, &gradients, perimeter, gamma](halfedge* e) {
            int source = e->source;
            int target = e->target;
            
            const float3& a = vertices[source];
            const float3& b = vertices[target];

            float3 delta = (b-a)/perimeter;
            gradients[source] -= gamma*delta;
            gradients[target] += gamma*delta;
        });
        return success;
    };
    bool perimeter_elasticity_grad(const vector<float3>& vertices,  vector<float3>& gradients, float k, float p0)
    {
        float perimeter = this->perimeter(vertices);
        if(perimeter < 0)
            return false;
        bool success = this->apply_edges([&vertices, &gradients, perimeter, k, p0](halfedge* e) {
            int source = e->source;
            int target = e->target;
            
            const float3& a = vertices[source];
            const float3& b = vertices[target];

            float3 delta = (b-a)/perimeter;
            gradients[source] -= k*delta*(perimeter-p0);
            gradients[target] += k*delta*(perimeter-p0);
        });
        return success;
    };
};


void halfedge_assign_prev(halfedge* edge, halfedge* new_prev)
{
    if(edge != nullptr)
    {
        halfedge* old_prev = edge->prev;
        edge->prev = new_prev;
        if(new_prev != nullptr)
        {
            new_prev->next = edge;
        }
    }
};
void halfedge_assign_next(halfedge* edge, halfedge* new_next)
{
    if(edge != nullptr)
    {
        halfedge* old_next = edge->next;
        edge->next = new_next;
        if(new_next != nullptr)
        {
            new_next->prev = edge;
        }
    }
};
void halfedge_assign_twins(halfedge* edge_1, halfedge* edge_2)
{
    if(edge_1 != nullptr && edge_2 != nullptr)
    {
        edge_1->twin = edge_2;
        edge_2->twin = edge_1;
    }
}
void halfedge_split(halfedge* old, int new_vertex_index)
{
    halfedge* old_twin = old->twin;
    halfedge* new_edges = (halfedge*)malloc(4*sizeof(halfedge));
    new_edges[0].source = old->source;
    new_edges[0].target = new_vertex_index;
    new_edges[1].source = new_vertex_index;
    new_edges[1].target = old->target;
    new_edges[2].source = old->target;
    new_edges[2].target = new_vertex_index;
    new_edges[3].source = new_vertex_index;
    new_edges[3].target = old->source;

    halfedge_assign_prev(&new_edges[0], old->prev);
    halfedge_assign_next(&new_edges[0], &new_edges[1]);
    halfedge_assign_next(&new_edges[1], old->next);
    if(old_twin != nullptr)
    {
        halfedge_assign_prev(&new_edges[2], old_twin->prev);
    }
    halfedge_assign_next(&new_edges[2], &new_edges[3]);
    if(old_twin != nullptr)
    {
        halfedge_assign_next(&new_edges[3], old_twin->next);
    }
    halfedge_assign_twins(&new_edges[0], &new_edges[3]);
    halfedge_assign_twins(&new_edges[1], &new_edges[2]);

    new_edges[0].parent = old->parent;
    new_edges[1].parent = old->parent;    
};
void halfedge_collapse(halfedge* old,  int new_vertex_index)
{

}


void halfedge_apply_neighbourhood(halfedge* edge,function<void(halfedge*)> func)
{
    func(edge);
        if(edge->twin != nullptr)
            func(edge->twin);
        halfedge* current = edge->next;
        halfedge* current_twin = current->twin;
        if(current_twin != nullptr)
        {
            while(true)
            {
                func(current);
                func(current_twin);

                
                current = current_twin->next;
                if(current == nullptr)
                    break;
                current_twin =  current->twin;
                if(current_twin == nullptr)
                    break;
                if(current_twin == edge)
                    break;
            };
        }
        else
        {
            func(current);
        }
        current = edge->prev;
        current_twin = current->twin;
        if(current_twin != nullptr)
        {
            while(true)
            {
                func(current);
                func(current_twin);
                current = current_twin->prev;
                if(current == nullptr)
                    break;
                current_twin =  current->twin;
                if(current_twin == nullptr)
                    break;
                if(current_twin == edge)
                    break;
            };
        }
        else
        {
            func(current);
        }
};

template<typename acc_type>
void halfedge_accumulate_target_vertex_neighbourhood(halfedge* edge,function<acc_type(int,int)> func, acc_type& value)
{
    
    int target = edge->target;
    halfedge* current = edge->next;
    halfedge* current_twin = current->twin;
    if(current_twin != nullptr)
    {
        while(true)
        {
            value += func(target, current->target);
            current = current_twin->next;
            if(current == nullptr)
                break;
            current_twin =  current->twin;
            if(current_twin == nullptr)
                break;
            if(current_twin == edge)
                break;
        };
    }
    else
    {
        value += func(target, current->target);
    };
};

template<typename acc_type>
void halfedge_accumulate_source_vertex_neighbourhood(halfedge* edge,function<acc_type(int,int)> func, acc_type& value)
{
    
    int source = edge->source;
    halfedge* current = edge->prev;
    halfedge* current_twin = current->twin;
    if(current_twin != nullptr)
    {
        while(true)
        {
            value += func(source, current->source);
            current = current_twin->prev;
            if(current == nullptr)
                break;
            current_twin =  current->twin;
            if(current_twin == nullptr)
                break;
            if(current_twin == edge)
                break;
        };
    }
    else
    {
        value += func(source, current->source);
    };
};

void load_polygons(vector<vector<int>> list_of_polygons, vector<polygon*>& polygons, vector<polygon*>& boundaries)
{
    polygons.reserve(list_of_polygons.size());

    polygon* pols = (polygon*)malloc(list_of_polygons.size()*sizeof(polygon));

    unordered_map<int2, int> edge_to_poly;
    unordered_map<int2, int> edge_to_index;
    unordered_map<int2,int2> edge_to_prev;
    unordered_map<int2,int2> edge_to_next;
    int polygon_index = 0;
    int halfedge_index = 0;
    map<int,int2> polygon_to_root; 
    for(vector<int> polygon : list_of_polygons)
    {
        pols[polygon_index].root = nullptr;
        polygons.push_back(&pols[polygon_index]);
        int n_sides = polygon.size();
        int2 prev_edge = {-1,-1};
        int2 first_edge;
        int2 last_edge;
        for(int i = 0; i < n_sides; i++)
        {
            int next_i = pmod((i+1), n_sides);
            int2 edge = {polygon[i], polygon[next_i]};
            if(i == 0)
                first_edge = edge;
            if(i == n_sides - 1)
                last_edge = edge;                
            edge_to_poly[edge] = polygon_index;
            edge_to_index[edge] = halfedge_index;
            if(i>0)
                edge_to_prev[edge] = prev_edge;
                edge_to_next[prev_edge] = edge;
            prev_edge = edge;
            halfedge_index += 1;
        };
        edge_to_prev[first_edge] = last_edge;
        edge_to_next[last_edge] = first_edge;
        polygon_to_root[polygon_index] = first_edge;
        polygon_index++;
    };
    
    int num_half_edges = edge_to_poly.size();
    halfedge* halfedges = (halfedge*)malloc(sizeof(halfedge)*num_half_edges);
    vector<int2> boundary_edges;
    for(auto const& edge_and_index : edge_to_index)
    {
        
        int2 edge = edge_and_index.first;
        int index = edge_and_index.second;
        int polygon_index = edge_to_poly[edge];
                
        halfedges[index].source = edge.x;
        halfedges[index].target = edge.y;
        int2 twin_edge = {edge.y, edge.x};
        int twin_index = -1;
        if(edge_to_index.count(twin_edge))
            twin_index = edge_to_index[twin_edge];
        if(twin_index != -1)
        {
            halfedges[index].twin = &halfedges[twin_index];
        }
        else
        {
            boundary_edges.push_back(edge);
        }
        int next_index =  edge_to_index[edge_to_next[edge]];
        int prev_index =  edge_to_index[edge_to_prev[edge]];
        halfedges[index].next = &halfedges[next_index];
        halfedges[index].prev = &halfedges[prev_index];
    };
    for(auto const& poly_and_edge : polygon_to_root)
    {
        polygons[poly_and_edge.first]->root = &halfedges[edge_to_index[poly_and_edge.second]];
    };
    
    vector<int2> external_edges;
    for(auto edge : boundary_edges)
    {
        external_edges.push_back({edge.y, edge.x});
    }

    int num_boundary_edges = external_edges.size();
    halfedge* external_halfedges = (halfedge*)malloc(sizeof(halfedge)*num_boundary_edges);
    unordered_map<int2,halfedge*> external_edge_to_halfedge;
    int external_halfedges_index = 0;
    for(int2 external_edge : external_edges)
    {
        int2 boundary_edge = {external_edge.y, external_edge.x};
        int index = edge_to_index[boundary_edge];
        halfedge& boundary_halfedge = halfedges[index];
        halfedge& external_halfedge = external_halfedges[external_halfedges_index];
        external_halfedge.twin = &boundary_halfedge;
        boundary_halfedge.twin = &external_halfedge;
        external_halfedge.source = boundary_halfedge.target;
        external_halfedge.target = boundary_halfedge.source;
        external_edge_to_halfedge[int2(external_halfedge.source, external_halfedge.target)] = &external_halfedge;
        external_halfedges_index += 1;
    };

    vector<vector<int2>> external_loops;
    vector<int2> current_loop;
    int2 current_edge;
    while(external_edges.size()>0)
    {
        if(current_loop.size() == 0)
        {
            current_edge = external_edges.front();
            external_edges.erase(external_edges.begin());
            current_loop.push_back(current_edge);
        }
        else
        {
            bool found_something = false;
            auto it = external_edges.begin();
            while (it != external_edges.end()) {
                if (it->x == current_edge.y) {
                    current_loop.push_back(*it);
                    current_edge = *it; // update current_edge to continue the chain
                    //cout << current_edge.first.x << " " << current_edge.first.y << endl;
                    it = external_edges.erase(it); // erase and get new iterator
                    found_something = true;
                    break;
                } else {
                    ++it;
                }
            };
            if(found_something == false)
            {
                external_loops.push_back(current_loop);
                current_loop.clear();
            };
        };
    };
    if(current_loop.size()>0)
    {
        external_loops.push_back(current_loop);
    };
    
    for(vector<int2> loop : external_loops)
    {
        halfedge* root = nullptr;
        int2 prev_edge;
        int2 first_edge;
        int2 last_edge;
        for(int i =0; i < loop.size(); i++)
        {
            int2 edge = loop[i];
            if(root == nullptr)
                root = external_edge_to_halfedge[edge];
            if(i == 0)
                first_edge = edge;
            if(i == loop.size()-1)
                last_edge = edge;
            if(i>0)
            {
                external_edge_to_halfedge[edge]->prev = external_edge_to_halfedge[prev_edge];
                external_edge_to_halfedge[prev_edge]->next = external_edge_to_halfedge[edge];
            };
            prev_edge = edge;
        };
        external_edge_to_halfedge[first_edge]->prev = external_edge_to_halfedge[last_edge];
        external_edge_to_halfedge[last_edge]->next = external_edge_to_halfedge[first_edge];
        polygon* loop_polygon = new polygon();
        loop_polygon->root = root;
        boundaries.push_back(loop_polygon);
    };
};

void create_quad_chain(int num_quads, float width, float height, vector<vector<int>>& polys, vector<float3>& vertices)
{
    int num_tot_vertices = 2*(num_quads+1);
    int num_upper_vertices = num_quads + 1;
    polys.reserve(num_quads);
    vertices.reserve(num_tot_vertices);
    for(int i = 0; i < num_tot_vertices; i++)
    {
        vertices.push_back({0.0f,0.0f,0.0f});
    }
    for(int i =0; i < num_quads; i++)
    {
        int upper_left = i;
        int upper_right = i + 1;
        int lower_left = i + num_quads + 1;
        int lower_right = i + num_quads + 2;
        polys.push_back({lower_left, lower_right, upper_right, upper_left});
        vertices[upper_left] = {i*width, height, 0.0f};
        vertices[lower_left] = {i*width, 0.0, 0.0f};
    }
    vertices[num_quads] = {num_quads*width, height, 0.0f};
    vertices[2*num_quads + 1] = {num_quads*width, 0.0, 0.0f};
}



