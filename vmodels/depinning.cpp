#include "hmath.h"
#include "helpers.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <fstream>
#include <set>
using namespace std;


class DepinningModel2d
{
    int Nx;
    int Ny;
    int N;
    real k0;
    real k;
    real disorder_mean;
    real disorder_std;
    real* f_drive;
    real* f_elastic;
    real* f_disorder;
    default_random_engine rng;
    normal_distribution<real> normal_dist;

    DepinningModel2d(int Nx_, int Ny_, real k0_, real k_, real disorder_mean_, real disorder_std_)
    {
        Nx = Nx_;
        Ny = Ny_;
        N = Nx*Ny;
        k0 = k0_;
        k = k_;
        disorder_mean = disorder_mean_;
        disorder_std = disorder_std_;

        f_drive = new real[Nx*Ny];
        f_elastic = new real[Nx*Ny];
        f_disorder = new real[Nx*Ny];
        active_sites = new int[Nx*Ny];
        num_active_sites = 0;

        normal_dist(disorder_mean, disorder_std);

        for(int i = 0; i < Nx*Ny; i++)
        {
            f_drive[i] = R(0.0);
            f_elastic[i] = R(0.0);
            f_disorder[i] = normal_dist(rng);
        }
    };


    int pack_index(int x, int y)
    {
        return y*Nx + x;
    };

    int unpack_index(int index, int& x, int& y)
    {
        x = index % Nx;
        y = index / Nx;
        return 0;
    };

    

    void propagate_avalanche(int epicenter)
    {
        set<int> sites;
        sites.insert(epicenter);
        set<int> touched_sites;
        set<int> avalanche_sites;
        set<int> new_sites;
        
        while(sites.size() > 0)
        {
            for(int site : sites)
            {
                
            }
        }

    };
}