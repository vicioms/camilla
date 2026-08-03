#include <cmath>
#include "hmath.h"
#include <iostream>
#include <fstream>
#include <random>
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

inline real overlap_interaction(
    const vec2& r_ij,
    const mat2& sigma_i,
    const mat2& sigma_j,
    const real det_sigma_i,
    const real det_sigma_j,
    vec2& r_ij_grad,
    mat2& sigma_i_grad,
    mat2& sigma_j_grad
)
{
    mat2 sigma_ij = sigma_i + sigma_j;

    real det_sigma_ij = det(sigma_ij);

    mat2 inv_sigma_ij = inverse_symm(sigma_ij);
    mat2 inv_sigma_i  = inverse_symm(sigma_i);
    mat2 inv_sigma_j  = inverse_symm(sigma_j);

    vec2 u = dot(inv_sigma_ij, r_ij);

    real quad_form =
        R(0.5) * dot(r_ij, u);

    real log_overlap =
        -quad_form
        + R(0.5) * std::log(det_sigma_i)
        + R(0.5) * std::log(det_sigma_j)
        - R(0.5) * std::log(det_sigma_ij);

    real overlap = std::exp(log_overlap);

    mat2 uuT = outer(u, u);

    r_ij_grad += -overlap * u;

    sigma_i_grad +=
        R(0.5) * overlap *
        (
            inv_sigma_i
            - inv_sigma_ij
            + uuT
        );

    sigma_j_grad +=
        R(0.5) * overlap *
        (
            inv_sigma_j
            - inv_sigma_ij
            + uuT
        );

    symmetrize(sigma_i_grad);
    symmetrize(sigma_j_grad);

    return overlap;
}

struct psystem
{
    int num_particles = 0;
    real cell_size = R(0.0);
    int2 cell_counts = int2(0, 0);
    int total_cells = 0;
    vec2 box_size = zero_vec2;
    bool use_pbc = true;
    default_random_engine rng;
    uniform_real_distribution<real> uniform_dist{R(0.0), R(1.0)};
    normal_distribution<real> normal_dist{R(0.0), R(1.0)};

    vec2* x = nullptr;
    mat2* sigma = nullptr;
    vec2* sigma_vals = nullptr;
    real* det_sigma = nullptr;
    real* trace_sigma = nullptr;
    vec2* x_grad = nullptr;
    mat2* sigma_grad = nullptr;
    int* x_cell_ids = nullptr;
    int* counts = nullptr;
    int* offsets = nullptr;
    int* index_list = nullptr;
    int* cell_cursors = nullptr;
    int* num_neighbors = nullptr;

    real k_A = R(0.0);
    real A0 = R(0.0);
    real k_P = R(0.0);
    real repulsion_strength = R(0.0);
    real lambda_max = R(0.0);
    
    psystem() = default;
    explicit psystem(int num_particles_, real cell_size_, int2 cell_counts_, real k_A_, real A0_, real k_P_, real repulsion_strength_, bool use_pbc_ = true)
    {
        num_particles = num_particles_;
        cell_size = cell_size_;
        cell_counts = cell_counts_;
        total_cells = cell_counts.x * cell_counts.y;
        box_size = vec2(cell_size*cell_counts.x, cell_size*cell_counts.y);
        use_pbc = use_pbc_;
        x = new vec2[num_particles];
        sigma = new mat2[num_particles];
        x_grad = new vec2[num_particles];
        sigma_grad = new mat2[num_particles];
        sigma_vals = new vec2[num_particles]; 
        det_sigma = new real[num_particles];
        trace_sigma = new real[num_particles];
        x_cell_ids = new int[num_particles];
        counts = new int[total_cells];
        offsets = new int[total_cells];
        index_list = new int[num_particles];
        cell_cursors = new int[total_cells];
        num_neighbors = new int[num_particles];
        k_A = k_A_;
        A0 = A0_;
        k_P = k_P_;
        repulsion_strength = repulsion_strength_;
    };
    psystem(const psystem&) = delete;
    psystem& operator=(const psystem&) = delete;

     ~psystem()
    {
        delete[] x;
        delete[] sigma;
        delete[] x_grad;
        delete[] sigma_grad;
        delete[] x_cell_ids;
        delete[] counts;
        delete[] offsets;
        delete[] index_list;
        delete[] cell_cursors;

        delete[] num_neighbors;
        delete[] sigma_vals;
        delete[] det_sigma;
        delete[] trace_sigma;
    }

    void init_lattice(real initial_area,  bool try_deterministic = true)
    {
        if (try_deterministic)
        {
            int expected_num_particles = cell_counts.x * cell_counts.y;
            if (num_particles != expected_num_particles)
            {
                std::cerr << "Warning: num_particles does not match expected number for deterministic lattice initialization. Expected: " << expected_num_particles << ", got: " << num_particles << ". Falling back to non-deterministic initialization." << std::endl;
                try_deterministic = false;
            }
        }
        if (try_deterministic)
        {
            for(int i = 0; i < num_particles; i++)
            {
                int x_idx = i % cell_counts.x;
                int y_idx = (i / cell_counts.x) % cell_counts.y;
                vec2 pos = vec2((x_idx + R(0.5)) * cell_size, (y_idx + R(0.5)) * cell_size);
                if(use_pbc)
                {
                    pos = wrap_box(pos, vec2(R(0.0), R(0.0)), box_size);
                }
                x[i] = pos;
                sigma[i] = mat2(initial_area*std::sqrt(uniform_dist(rng) + 1.0), R(0.0), R(0.0), initial_area*std::sqrt(uniform_dist(rng) + 1.0));
            };
        }
        else
        {

            for(int i = 0; i < num_particles; i++)
            {
                x[i] = vec2(uniform_dist(rng) * box_size.x, uniform_dist(rng) * box_size.y);
                sigma[i] = mat2(initial_area*std::sqrt(uniform_dist(rng) + 1.0), R(0.0), R(0.0), initial_area*std::sqrt(uniform_dist(rng) + 1.0));
            };
        }
        
    }

    void update_cell_list()
    {
        lambda_max = R(0.0);

        for(int c = 0; c < total_cells; c++)
        {
            counts[c] = 0;
            offsets[c] = 0;
            cell_cursors[c] = 0;
        }

        for(int i = 0; i < num_particles; i++)
        {
            index_list[i] = -1;

            int cell_x = static_cast<int>(std::floor(x[i].x / cell_size));
            int cell_y = static_cast<int>(std::floor(x[i].y / cell_size));

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

            int cell_id = get_cell_id(cell_x, cell_y, cell_counts.x, cell_counts.y);

            x_cell_ids[i] = cell_id;
            counts[cell_id]++;
        }

        offsets[0] = 0;

        for(int c = 1; c < total_cells; c++)
        {
            offsets[c] = offsets[c - 1] + counts[c - 1];
        }

        for(int i = 0; i < num_particles; i++)
        {
            int cell_id = x_cell_ids[i];
            int k = offsets[cell_id] + cell_cursors[cell_id];

            index_list[k] = i;
            cell_cursors[cell_id]++;
        }
    };

    void update_loop()
    {
        lambda_max = R(0.0);

        for(int i = 0; i < num_particles; i++)
        {
            num_neighbors[i] = 0;
            x_grad[i] = zero_vec2;
            sigma_grad[i] = zero_mat2;
            spectral_data_symm(
                sigma[i],
                trace_sigma[i],
                det_sigma[i],
                sigma_vals[i]
            );

            lambda_max = std::max(
                lambda_max,
                std::max(sigma_vals[i].x, sigma_vals[i].y)
            );


            real A = std::sqrt(det_sigma[i]);

            mat2 cof_sigma_i = cofactor(sigma[i]);

            sigma_grad[i] +=
                R(0.5) * k_A * (A - A0) * cof_sigma_i / A;

            mat2 inv_sigma_i = inverse_symm(sigma[i]);
            mat2 inv_sqrt_sigma_i = inverse_sqrt_symm(sigma[i]);

            real tr_inv_sqrt =
                inv_sqrt_sigma_i.m00 + inv_sqrt_sigma_i.m11;

            mat2 inv_3_2_sigma_i = matmul(
                inv_sqrt_sigma_i,
                inv_sigma_i
            );

            symmetrize(inv_3_2_sigma_i);

            sigma_grad[i] +=
                R(0.5) * k_P * A *
                (
                    tr_inv_sqrt * inv_sigma_i
                    - inv_3_2_sigma_i
                );

            symmetrize(sigma_grad[i]);
        };

        real maximal_cutoff2 = R(16.0) * lambda_max;

        int cell_range = static_cast<int>(
            std::ceil(std::sqrt(maximal_cutoff2) / cell_size)
        );

        

        if(use_pbc)
        {
            if(2 * cell_range + 1 > cell_counts.x ||
               2 * cell_range + 1 > cell_counts.y)
            {
                throw std::runtime_error(
                    "cell_range too large for PBC: duplicate wrapped neighbor cells possible"
                );
            }
        }

        for(int c = 0; c < total_cells; c++)
        {
            int start = offsets[c];
            int end = offsets[c] + counts[c];

            int2 c_coords = get_cell_coords(c, cell_counts);

            for(int c_dy = -cell_range; c_dy <= cell_range; c_dy++)
            {
                for(int c_dx = -cell_range; c_dx <= cell_range; c_dx++)
                {
                    int2 neighbor_coords = int2(
                        c_coords.x + c_dx,
                        c_coords.y + c_dy
                    );

                    if(use_pbc)
                    {
                        neighbor_coords.x = pmod(neighbor_coords.x, cell_counts.x);
                        neighbor_coords.y = pmod(neighbor_coords.y, cell_counts.y);
                    }
                    else
                    {
                        if(neighbor_coords.x < 0 || neighbor_coords.x >= cell_counts.x ||
                           neighbor_coords.y < 0 || neighbor_coords.y >= cell_counts.y)
                        {
                            continue;
                        }
                    }

                    int neighbor_cell_id = get_cell_id(neighbor_coords, cell_counts);

                    int neighbor_start = offsets[neighbor_cell_id];
                    int neighbor_end = offsets[neighbor_cell_id] + counts[neighbor_cell_id];

                    for(int p_i = start; p_i < end; p_i++)
                    {
                        int i = index_list[p_i];
                        vec2 r_i = x[i];

                        for(int p_j = neighbor_start; p_j < neighbor_end; p_j++)
                        {
                            int j = index_list[p_j];

                            if(j >= i)
                                continue;

                            vec2 r_j = x[j];

                            vec2 r_ij;
                            if(use_pbc)
                                r_ij = wrap_diff(r_i, r_j, box_size);
                            else
                                r_ij = r_i - r_j;

                            real r2_ij = dot(r_ij, r_ij);

                            if(r2_ij > maximal_cutoff2)
                                continue;


                            vec2 r_ij_grad = zero_vec2;
                            mat2 sigma_i_grad = zero_mat2;
                            mat2 sigma_j_grad = zero_mat2;
                            real overlap = overlap_interaction(
                                r_ij,
                                sigma[i],
                                sigma[j],
                                det_sigma[i],
                                det_sigma[j],
                                r_ij_grad,
                                sigma_i_grad,
                                sigma_j_grad
                            );

                            x_grad[i] += repulsion_strength*r_ij_grad;
                            x_grad[j] -=  repulsion_strength*r_ij_grad;
                            sigma_grad[i] += repulsion_strength*sigma_i_grad;
                            sigma_grad[j] += repulsion_strength*sigma_j_grad;

                            num_neighbors[i]++;
                            num_neighbors[j]++;
                        }
                    }
                }
            }
        }
    }

    void step(real dt, real translational_diffusion, real rotational_diffusion)
    {
        for(int i = 0; i < num_particles; i++)
        {
            x[i] -= dt * x_grad[i];
            if (translational_diffusion > R(0.0))
            {
                x[i] += std::sqrt(2.0 * translational_diffusion * dt) * vec2(normal_dist(rng), normal_dist(rng));
            }
            sigma[i] -= dt * sigma_grad[i];
            symmetrize(sigma[i]);
            if (rotational_diffusion > R(0.0))
            {
                rotate(sigma[i], std::sqrt(2.0 * rotational_diffusion * dt) * normal_dist(rng));
            };
            symmetrize(sigma[i]);
            if(use_pbc)
            {
                x[i] = wrap_box(x[i], vec2(R(0.0), R(0.0)), box_size);
            }
        }
    }
};

int main()
{
    int2 cell_counts = int2(40, 40);
    int num_particles = cell_counts.x * cell_counts.y;

    real cell_size = R(1.0);

    real A0 = R(0.5);
    real initial_area = A0;

    real k_A = R(12.0);
    real k_P = R(0.1);

    real repulsion_strength = R(1.0);

    real D = R(0.01);
    real D_rot = R(0.01);

    bool use_pbc = true;

    real dt = R(0.001);

    psystem particle_sys(
        num_particles,
        cell_size,
        cell_counts,
        k_A,
        A0,
        k_P,
        repulsion_strength,
        use_pbc
    );
    particle_sys.init_lattice(initial_area);
    std::ofstream file("dump/deformable.bin", std::ios::binary);
    for(int step = 0; step < 20000; step++)
    {
        if(step % 100 == 0)
        {
            cout << "Step " << step << endl;
        }
        
        particle_sys.update_cell_list();
        particle_sys.update_loop();
        particle_sys.step(dt, D, D_rot);
        for(int i = 0; i < num_particles; i++)
        {
            file.write(reinterpret_cast<const char*>(&particle_sys.x[i].x), sizeof(real));
            file.write(reinterpret_cast<const char*>(&particle_sys.x[i].y), sizeof(real));
            file.write(reinterpret_cast<const char*>(&particle_sys.sigma[i].m00), sizeof(real));
            file.write(reinterpret_cast<const char*>(&particle_sys.sigma[i].m01), sizeof(real));
            file.write(reinterpret_cast<const char*>(&particle_sys.sigma[i].m10), sizeof(real));
            file.write(reinterpret_cast<const char*>(&particle_sys.sigma[i].m11), sizeof(real));
        };
        file.flush();
    };
    
    file.close();
}