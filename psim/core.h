#pragma once
#include "../generic.h"

struct System
{
    float3 domain_min;
    float3 domain_max;
    int3 grid_size;
    float cell_size;
    float cutoff;
    bool use_periodic;
};



