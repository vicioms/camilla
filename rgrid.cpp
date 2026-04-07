#include <iostream>
#include <random>
#include <cmath>
#include <vector>
using namespace std;

inline int get_bit(int x, int n) {
    return (x >> n) & 1;
}
inline int set_bit(int x, int n) {
    return x | (1 << n);
}
inline int unset_bit(int x, int n) {
    return x & ~(1 << n);
}

struct halfgrid45
{
    int size;
    int num_nodes;
    int num_edges;
    float* node_data;
    float* edge_data;

    // l = 0, offset = 0
    // l = 1, offset = 1
    // l = 2, offset = 3
    // offset = l * (l + 1) / 2
    int _level_offset_node(int level)
    {
        return level * (level + 1) / 2;
    };

    int _level_offset_edge(int level)
    {
        return level * (level + 1);
    };

    int _num_nodes(int level)
    {
        return level + 1;
    };

    int _num_out_edges(int level)
    {
        return 2 * (level + 1);  
    };

    int _node_index(int level, int n)
    {
        return _level_offset_node(level) + n;
    };

    int _edge_index(int level, int n, bool right)
    {
        return _level_offset_edge(level) + n * 2 + (right ? 1 : 0);
    };

    float _path_edge_sum(int level, int n, const vector<bool>& path)
    {
        float sum = 0.0f;
        int current_node = n;
        for(int l_idx = 0; l_idx < (int)path.size(); l_idx++)
        {
            int l = level + l_idx;
            int edge_index = _edge_index(l, current_node, path[l_idx]); // was n, should be current_node
            if (edge_index >= num_edges) break;
            sum += edge_data[edge_index];
            current_node = path[l_idx] ? current_node + 1 : current_node;
        }
        return sum;
    };

    float _djikstra(int start_level, int start_node, vector<bool>& path)
    {
        // Implement Dijkstra's algorithm to find the shortest path from (start_level, start_node)
        float sum = 0.0f;
        int node = start_node;
        for(int level = start_level; level < size; level++)
        {
            int edge_offset = _level_offset_edge(level);
            float left_edge_weight = edge_data[edge_offset + node * 2];
            float right_edge_weight = edge_data[edge_offset + node * 2 + 1];
            // Update path based on edge weights and continue to next level
            if (left_edge_weight < right_edge_weight) {
                sum += left_edge_weight;
                path[level - start_level] = false; // choose left edge
                node = node; // stay on the same node index for left edge
            } else {
                sum += right_edge_weight;
                path[level - start_level] = true; // choose right edge
                node = node + 1; // move to the next node index for right edge
            }
        };
        return sum;
    };


    halfgrid45(int size) : size(size) {
        num_nodes = size * (size + 1) / 2;
        num_edges = size * (size + 1);
        node_data = new float[num_nodes];
        edge_data = new float[num_edges];
    };

};


int main()
{
    default_random_engine generator;
    generator.seed(42); // Set a fixed seed for reproducibility
    normal_distribution<float> distribution(0.0f, 1.0f);
    int size = 5; // Example size
    halfgrid45 grid(size);
    for(int i = 0; i < grid.num_edges; i++) {
        grid.edge_data[i] = distribution(generator);
    };

    for(int l = 0; l < grid.size; l++) {
        cout << "Level " << l << ": ";
        for(int n = 0; n < grid._num_nodes(l); n++) {
            cout << "(" << grid._node_index(l, n) << ") ";
                int edge_index_left = grid._edge_index(l, n, false);
                int edge_index_right = grid._edge_index(l, n, true);
                if (edge_index_left < grid.num_edges) {
                    cout << "L:" << grid.edge_data[edge_index_left] << " ";
                }
                if (edge_index_right < grid.num_edges) {
                    cout << "R:" << grid.edge_data[edge_index_right] << " ";
                }
        }
        cout << endl;
    }

    vector<bool> best_path(grid.size);
    float best_sum = grid._djikstra(0, 0, best_path);
    cout << "Best path from (0, 0): " << best_sum << endl;
    for(int i = 0; i < best_path.size(); i++) {
        cout << best_path[i];
    }
    cout << endl;
    cout << "Path edge sum: " << grid._path_edge_sum(0, 0, best_path) << endl;
    return 0;
}
