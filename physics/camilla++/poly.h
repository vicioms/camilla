#pragma once
#include "../generic.h"
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


struct polymesh
{
    vector<float3> vertices;
    vector<vector<int>> polygons;
    vector<vector<int2>> twinPolyEdges;

    //example of data 
    // 0: [0,1,2]
    // 1: [4, 3,2,1]

    // 0: [(-1,-1),(1, 2),(-1,-1)] 
    // 1: [(-1,-1),(-1-1),(0,1), (-1,-1)]


};

