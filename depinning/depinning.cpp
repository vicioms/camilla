#include<vector>
#include<functional>
#include<unordered_map>
#include<fstream>
#include<random>
#include<limits>
#include<set>
#include<string>
#include<iostream>
#include<sstream>
#include "../argparser.h"
using namespace std;
static constexpr float inf = std::numeric_limits<float>::infinity();

void load_snapshot(const std::string& filename,
                   int& Nx, int& Ny,
                   float*& f_el,
                   float*& f_drive,
                   float*& f_friction,
                   std::default_random_engine& rng,
                   float& dh, float& k0, float& k,
                   float& fric_m, float& fric_s)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open snapshot file for reading");

    in.read((char*)&Nx, sizeof(int));
    in.read((char*)&Ny, sizeof(int));
    in.read((char*)&dh, sizeof(float));
    in.read((char*)&k0, sizeof(float));
    in.read((char*)&k, sizeof(float));
    in.read((char*)&fric_m, sizeof(float));
    in.read((char*)&fric_s, sizeof(float));

    size_t N = Nx * Ny;
    f_el = new float[N];
    f_drive = new float[N];
    f_friction = new float[N];

    in.read((char*)f_el, sizeof(float) * N);
    in.read((char*)f_drive, sizeof(float) * N);
    in.read((char*)f_friction, sizeof(float) * N);

    size_t len;
    in.read((char*)&len, sizeof(size_t));
    std::string rng_str(len, '\0');
    in.read(&rng_str[0], len);
    std::istringstream rng_ss(rng_str);
    rng_ss >> rng;

    in.close();
}

void save_snapshot(const std::string& filename,
                   int Nx, int Ny,
                   const float* f_el,
                   const float* f_drive,
                   const float* f_friction,
                   const std::default_random_engine& rng,
                   float dh, float k0, float k,
                   float fric_m, float fric_s)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open snapshot file for writing");

    out.write((char*)&Nx, sizeof(int));
    out.write((char*)&Ny, sizeof(int));
    out.write((char*)&dh, sizeof(float));
    out.write((char*)&k0, sizeof(float));
    out.write((char*)&k, sizeof(float));
    out.write((char*)&fric_m, sizeof(float));
    out.write((char*)&fric_s, sizeof(float));
    out.write((char*)f_el, sizeof(float) * Nx * Ny);
    out.write((char*)f_drive, sizeof(float) * Nx * Ny);
    out.write((char*)f_friction, sizeof(float) * Nx * Ny);

    std::ostringstream rng_ss;
    rng_ss << rng;
    std::string rng_str = rng_ss.str();
    size_t len = rng_str.size();
    out.write((char*)&len, sizeof(size_t));
    out.write(rng_str.data(), len);

    out.close();
}

inline int pack_int2(int x, int y, int N) {
    return y * N + x;
}

inline int pmod(int i, int n) {
    return (i % n + n) % n;
}

inline void add(float* u, float value, int n) {
    for (int i = 0; i < n; i++) u[i] += value;
}

inline void zero(float* u, int n) {
    for (int i = 0; i < n; i++) u[i] = 0.0f;
}

float draw_disorder(default_random_engine& eng, normal_distribution<float>& distr) {
    return abs(distr(eng));
}

void propagate(float* f_el, float* f_drive, float* f_friction,
               default_random_engine& eng,
               normal_distribution<float>& distr,
               float dh, float k0, float k,
               set<int>& active_sites, set<int>& updated_sites,
               bool periodic_x, bool periodic_y,
               int Nx, int Ny)
{
    for (int i : active_sites) {
        updated_sites.insert(i);
        int x = i % Nx;
        int y = i / Nx;
        f_drive[i] -= k0 * dh;
        f_el[i] -= 4 * k * dh;
        f_friction[i] = draw_disorder(eng, distr);

        auto neighbor = [&](int xi, int yi) {
            int idx = pack_int2(xi, yi, Nx);
            f_el[idx] += k * dh;
            updated_sites.insert(idx);
        };

        if (periodic_x || x > 0) neighbor(pmod(x - 1, Nx), y);
        if (periodic_x || x < Nx - 1) neighbor(pmod(x + 1, Nx), y);
        if (periodic_y || y > 0) neighbor(x, pmod(y - 1, Ny));
        if (periodic_y || y < Ny - 1) neighbor(x, pmod(y + 1, Ny));
    }
}

void find_epicenter(float* f_el, float* f_drive, float* f_friction, int Nx, int Ny, float& min_failure_dist, int& epicenter) {
    min_failure_dist = inf;
    epicenter = -1;
    for (int i = 0; i < Nx * Ny; i++) {
        float delta = f_friction[i] - f_el[i] - f_drive[i];
        if (delta < min_failure_dist) {
            epicenter = i;
            min_failure_dist = delta;
        }
    }
}

class depinning {
public:
    int Nx, Ny;
    float dh, k0, k;
    float fric_m, fric_s;
    float* f_el;
    float* f_drive;
    float* f_friction;
    std::default_random_engine eng;
    normal_distribution<float> distr;

    depinning(int Nx_, int Ny_, float dh_, float k0_, float k_, float fric_m_, float fric_s_, int seed)
        : Nx(Nx_), Ny(Ny_), dh(dh_), k0(k0_), k(k_), fric_m(fric_m_), fric_s(fric_s_),
          distr(fric_m_, fric_s_) {
        eng.seed(seed);
        f_el = new float[Nx * Ny];
        f_drive = new float[Nx * Ny];
        f_friction = new float[Nx * Ny];
        zero(f_el, Nx * Ny);
        zero(f_drive, Nx * Ny);
        for (int i = 0; i < Nx * Ny; i++) f_friction[i] = draw_disorder(eng, distr);
    };

    void save(const std::string& filename) {
        save_snapshot(filename, Nx, Ny, f_el, f_drive, f_friction, eng, dh, k0, k, fric_m, fric_s);
    };

    void load(const std::string& filename) {
        delete[] f_el;
        delete[] f_drive;
        delete[] f_friction;
        load_snapshot(filename, Nx, Ny, f_el, f_drive, f_friction, eng, dh, k0, k, fric_m, fric_s);
        distr = normal_distribution<float>(fric_m, fric_s);
    };

    depinning(const std::string& filename)
    {
        load(filename);
    };

    ~depinning() {
        delete[] f_el;
        delete[] f_drive;
        delete[] f_friction;
    }

    

    void run(int num_steps, const std::string& output_file, bool periodic_x = true, bool periodic_y = true) {
        set<int> active_sites, updated_sites;
        vector<int> av_sizes;
        av_sizes.reserve(num_steps);

        for (int step = 0; step < num_steps; step++) {
            if (step % 1000 == 0) cout << step << endl;

            float min_failure_dist;
            int epicenter;
            find_epicenter(f_el, f_drive, f_friction, Nx, Ny, min_failure_dist, epicenter);
            add(f_drive, min_failure_dist, Nx * Ny);
            active_sites.insert(epicenter);
            int av_size = 1;

            while (!active_sites.empty()) {
                propagate(f_el, f_drive, f_friction, eng, distr, dh, k0, k,
                          active_sites, updated_sites, periodic_x, periodic_y, Nx, Ny);
                av_size += active_sites.size();
                active_sites.clear();

                for (int i : updated_sites) {
                    if (f_friction[i] < f_el[i] + f_drive[i]) {
                        active_sites.insert(i);
                    }
                }
                updated_sites.clear();
            }
            av_sizes.push_back(av_size);
        }
        if(output_file != "")
        {
            ofstream file(output_file);
            for (int s : av_sizes) file << s << '\n';
            file.close();
        } 
    }
};

int main(int argc, char** argv)
{
    ArgParser parser(argc,argv);
    int Nx;
    int Ny;
    float dh;
    float k0;
    float k;
    float fric_m;
    float fric_s;
    int seed;
    depinning* dep;

    if (parser.has("isnap")) {
        std::string snapshot = parser.get("isnap");
        dep = new depinning(snapshot);
    } else {
        Nx = parser.get_int("Nx", 128);
        Ny = parser.get_int("Ny", 128);
        dh = parser.get_float("dh", 0.1f);
        k0 = parser.get_float("k0", 0.01f);
        k  = parser.get_float("k", 1.0f);
        fric_m = parser.get_float("fric_m", 0.0f);
        fric_s = parser.get_float("fric_s", 1.0f);
        seed = parser.get_int("seed", 42);
        dep = new depinning(Nx, Ny, dh, k0, k, fric_m, fric_s, seed);
    }
    dep->run(parser.get_int("steps",1000000), parser.get("output",""));
    if(parser.get("fsnap","") != "")
    {
        dep->save(parser.get("fsnap"));
    };
}