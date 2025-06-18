#pragma once
#include "base.h"
#include <vector>
#include <functional>
#include <unordered_map>
using namespace std;

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
    }
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
    //A = 0.5*|| \sum_i v_i x v_{i+1} ||
    bool area_grad(const vector<float3>& vertices,  vector<float3>& gradients)
    {
        float3 area_v = this->area_vector(vertices);
        if(area_v == nan3)
            return false;
        bool success = this->apply_edges([&vertices, &gradients, area_v](halfedge* e) {
            int source = e->source;
            int target = e->target;
            float3 source_grad;
            float3 target_grad;
            gradients[source] += source_grad;
            gradients[target] += target_grad;
    
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
};
