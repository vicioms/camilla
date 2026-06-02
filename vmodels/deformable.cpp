#include <cmath>
#include "hmath.h"
#include <iostream>
#include <fstream>

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

inline int cell_id(int x, int y, int n_x, int n_y)
{
    return y * n_x + x;
}
inline int cell_id(const int2 x, const int2 cell_counts)
{
    return cell_id(x.x, x.y, cell_counts.x, cell_counts.y);
}
inline int cell_id(int x, int  y, int z, int n_x, int n_y, int n_z)
{
    return z * n_x * n_y + y * n_x + x;
}
inline int cell_id(const int3 x, const int3 cell_counts)
{
    return cell_id(x.x, x.y, x.z, cell_counts.x, cell_counts.y, cell_counts.z);
}
inline int2 cell_coords(int cell_id, int n_x, int n_y)
{
    int y = cell_id / n_x;
    int x = cell_id % n_x;
    return int2(x, y);
};
inline int2 cell_coords(int cell_id, const int2 cell_counts)
{
    return cell_coords(cell_id, cell_counts.x, cell_counts.y);
};
inline int3 cell_coords(int cell_id, int n_x, int n_y, int n_z)
{
    int z = cell_id / (n_x * n_y);
    int y = (cell_id % (n_x * n_y)) / n_x;
    int x = cell_id % n_x;
    return int3(x, y, z);
};
inline int3 cell_coords(int cell_id, const int3 cell_counts)
{
    return cell_coords(cell_id, cell_counts.x, cell_counts.y, cell_counts.z);
};

struct psystem
{
    int num_particles = 0;
    real cell_size = R(0.0);
    int2 cell_counts = int2(0, 0);
    bool use_pbc = true;
    vec2* x = nullptr;
    mat2* sigma = nullptr;
    vec2* x_grad = nullptr;
    mat2* sigma_grad = nullptr;
    int* x_cell_ids = nullptr;
    int num_occupied_cells = 0;
    int* occupied_cells = nullptr;
    
    psystem() = default;
    explicit psystem(int num_particles_, real cell_size_, int2 cell_counts_, bool use_pbc_ = true)
    {
        num_particles = num_particles_;
        cell_size = cell_size_;
        cell_counts = cell_counts_;
        use_pbc = use_pbc_;
        x = new vec2[num_particles];
        sigma = new mat2[num_particles];
        x_grad = new vec2[num_particles];
        sigma_grad = new mat2[num_particles];
        x_cell_ids = new int[num_particles];
        num_occupied_cells = 0;
        occupied_cells = new int[cell_counts.x * cell_counts.y];
    };

    void update_cell_ids()
    {
        for(int i = 0; i < num_particles; i++)
        {
            int cell_x = static_cast<int>(x[i].x / cell_size);
            int cell_y = static_cast<int>(x[i].y / cell_size);
            if (use_pbc)
            {
                cell_x = pmod(cell_x, cell_counts.x);
                cell_y = pmod(cell_y, cell_counts.y);
            }
            else
            {
                if (cell_x < 0) cell_x = 0;
                if (cell_x >= cell_counts.x) cell_x = cell_counts.x - 1;
                if (cell_y < 0) cell_y = 0;
                if (cell_y >= cell_counts.y) cell_y = cell_counts.y - 1;
            }
            x_cell_ids[i] = cell_id(cell_x, cell_y, cell_counts.x, cell_counts.y);
        };
    };

};

