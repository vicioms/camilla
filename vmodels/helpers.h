#pragma once
#include "hmath.h"
#include <cmath>
#include <random>
#include <vector>
using namespace std;


int digitize(
    real* x,
    int n,
    int num_bins,
    real min_val,
    real max_val,
    int* bin_x,
    int* bin_counts,
    int& left_out_of_range,
    int& right_out_of_range
)
{
    left_out_of_range = 0;
    right_out_of_range = 0;

    for (int i = 0; i < n; i++)
        bin_x[i] = -1;

    for (int b = 0; b < num_bins; b++)
        bin_counts[b] = 0;

    // Safety checks
    if (n < 0) return -1;
    if (num_bins <= 0) return -1;
    if (max_val <= min_val) return -1;

    real bin_width = (max_val - min_val) / static_cast<real>(num_bins);

    if (bin_width <= 0.0)
        return -1;

    for (int i = 0; i < n; i++)
    {
        int bin_index =
            static_cast<int>(std::floor((x[i] - min_val) / bin_width));

        if (bin_index < 0)
        {
            bin_x[i] = -1;
            left_out_of_range++;
            continue;
        }

        if (bin_index >= num_bins)
        {
            bin_x[i] = num_bins;
            right_out_of_range++;
            continue;
        }

        bin_x[i] = bin_index;
        bin_counts[bin_index]++;
    }

    return 0;
}

int digitize_log(
    real* x,
    int n,
    int num_bins,
    real min_exp,
    real max_exp,
    real base,
    real scale,
    int* bin_x,
    int* bin_counts,
    int& left_out_of_range,
    int& right_out_of_range
)
{
    left_out_of_range = 0;
    right_out_of_range = 0;

    for (int i = 0; i < n; i++)
        bin_x[i] = -1;

    for (int b = 0; b < num_bins; b++)
        bin_counts[b] = 0;

    if (n < 0) return -1;
    if (num_bins <= 0) return -1;
    if (base <= 0.0 || base == 1.0) return -1;
    if (scale <= 0.0) return -1;
    if (min_exp <= 0.0 || max_exp <= 0.0) return -1;
    if (max_exp <= min_exp) return -1;

    real log_base = std::log(base);

    real log_min = std::log(min_exp) / log_base;
    real log_max = std::log(max_exp) / log_base;

    real bin_width = (log_max - log_min) / static_cast<real>(num_bins);

    if (bin_width <= 0.0)
        return -1;

    for (int i = 0; i < n; i++)
    {
        if (x[i] <= 0.0)
        {
            left_out_of_range++;
            continue;
        }

        real y = x[i] / scale;

        if (y <= 0.0)
        {
            left_out_of_range++;
            continue;
        }

        real log_y = std::log(y) / log_base;

        int bin_index =
            static_cast<int>(std::floor((log_y - log_min) / bin_width));

        if (bin_index < 0)
        {
            bin_x[i] = -1;
            left_out_of_range++;
            continue;
        }

        if (bin_index >= num_bins)
        {
            bin_x[i] = num_bins;
            right_out_of_range++;
            continue;
        }

        bin_x[i] = bin_index;
        bin_counts[bin_index]++;
    }

    return 0;
}

inline int get_cell_id(int x, int y, int n_x, int n_y)
{
    return y * n_x + x;
}
inline int get_cell_id(const int2 x, const int2 cell_counts)
{
    return get_cell_id(x.x, x.y, cell_counts.x, cell_counts.y);
}
inline int get_cell_id(int x, int  y, int z, int n_x, int n_y, int n_z)
{
    return z * n_x * n_y + y * n_x + x;
}
inline int get_cell_id(const int3 x, const int3 cell_counts)
{
    return get_cell_id(x.x, x.y, x.z, cell_counts.x, cell_counts.y, cell_counts.z);
}
inline int2 get_cell_coords(int cell_id, int n_x, int n_y)
{
    int y = cell_id / n_x;
    int x = cell_id % n_x;
    return int2(x, y);
};
inline int2 get_cell_coords(int cell_id, const int2 cell_counts)
{
    return get_cell_coords(cell_id, cell_counts.x, cell_counts.y);
};
inline int3 get_cell_coords(int cell_id, int n_x, int n_y, int n_z)
{
    int z = cell_id / (n_x * n_y);
    int y = (cell_id % (n_x * n_y)) / n_x;
    int x = cell_id % n_x;
    return int3(x, y, z);
};
inline int3 get_cell_coords(int cell_id, const int3 cell_counts)
{
    return get_cell_coords(cell_id, cell_counts.x, cell_counts.y, cell_counts.z);
};

struct hgrid
{
    vector<uint32_t> particle_indices;
    vector<uint64_t> cell_keys;
    vector<size_t> cell_ptr;
    vector<size_t> level_ptr;


    
}
