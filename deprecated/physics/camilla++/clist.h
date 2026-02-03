#pragma once
#include "../generic.h"
#include <vector>
#include <functional>
using namespace std;

class clist2
{
private:
    float2 cell_size;
    int2 num_cells;
    bool periodic_x;
    bool periodic_y;
    float2 system_size;
    vector<int> point_indices;
    vector<int> cell_indices;
    vector<int> cell_start;
    vector<int> cell_end;

    int2 find_cell(const float2& p)
    {
        int x;
        int y;
        if(periodic_x)
        {
           x = (int)floor(fpmod(p.x, system_size.x)/cell_size.x);
        }
        else
        {
            x = (int)floor(p.x/cell_size.x);
        }
        if(periodic_y)
        {
           y = (int)floor(fpmod(p.y, system_size.y)/cell_size.y);
        }
        else
        {
            y = (int)floor(p.y/cell_size.y);
        }
        return int2(x,y);
    };
    int2 shift_cell(const int2& c, int dx, int dy)
    {
        int2 res;
        res.x = c.x + dx;
        res.y = c.y + dy;
        if(periodic_x)
        {
            res.x = pmod(res.x, num_cells.x);
        }
        if(periodic_y)
        {
            res.y = pmod(res.y, num_cells.y);
        };
        return res;
    }
    int pack_cell(const int2& c)
    {
        return c.y*num_cells.x + c.x;
    }

public:
    clist2(float cell_size_, int num_cells_, bool periodic = true)
    {
        periodic_x = periodic;
        periodic_y = periodic;
        cell_size = float2(cell_size_, cell_size_);
        num_cells = int2(num_cells_, num_cells_);
        system_size = float2(cell_size_*num_cells_, cell_size_*num_cells_);
    };
    clist2(float cell_size_, int num_cells_x, int num_cells_y, bool periodic = true)
    {
        periodic_x = periodic;
        periodic_y = periodic;
        cell_size = float2(cell_size_, cell_size_);
        num_cells = int2(num_cells_x, num_cells_y);
        system_size = float2(cell_size_*num_cells_x, cell_size_*num_cells_y);
    };

    

    void clear()
    {
        point_indices.clear();
        cell_indices.clear();
        cell_start.clear();
        cell_end.clear();
    };

    void build(vector<float2>& points)
    {
        clear();
        int num_points = points.size();
        point_indices.resize(num_points);
        cell_indices.resize(num_points);
        
        for (int i = 0; i < num_points; ++i) {
            int2 cell = find_cell(points[i]);
            int cell_index = pack_cell(cell); 
            point_indices[i] = i;
            cell_indices[i] = cell_index;       
        };

        // Sort point_indices by cell_indices
        std::sort(point_indices.begin(), point_indices.end(),
              [&](int a, int b) {
                  return cell_indices[a] < cell_indices[b];
              });
        vector<int> sorted_cell_indices(num_points);
        for (int i = 0; i < num_points; ++i)
            sorted_cell_indices[i] = cell_indices[point_indices[i]];

        int total_cells = num_cells.x * num_cells.y;
        cell_start.assign(total_cells, -1);
        cell_end.assign(total_cells, -1);
        // Compute cell_start and cell_end
        for (int i = 0; i < num_points; ++i) {
            int cell = sorted_cell_indices[i];
            if (cell_start[cell] == -1)
                cell_start[cell] = i;
            cell_end[cell] = i + 1;
        }
        cell_indices = std::move(sorted_cell_indices);
    };

    template<typename pointType>
    void loop_neighbors(vector<pointType>& objects, function<float2&(pointType&)> converter, function<void (float2&, float2&, clist2&)> func)
    {
        for(int i = 0; i < objects.size(); i++)
        {
            int2 cell = find_cell(converter(objects[i]));
            for(int dx = -1; dx <= 1; dx++)
            {
                for(int dy = -1; dy <= 1; dy++)
                {
                    int2 neighbor_cell = shift_cell(cell, dx, dy);
                    if ((!periodic_x && (neighbor_cell.x < 0 || neighbor_cell.x >= num_cells.x)) ||
                        (!periodic_y && (neighbor_cell.y < 0 || neighbor_cell.y >= num_cells.y)))
                        continue;
                    int neighbor_index = pack_cell(neighbor_cell);
                    int ptr_start = cell_start[neighbor_index];
                    if (ptr_start == -1) continue;
                    int ptr_end = cell_end[neighbor_index];
                    for(int ptr = ptr_start; ptr < ptr_end; ptr++)
                    {
                        int j = point_indices[ptr];
                        if(i == j)
                        {
                            continue;
                        }
                        func(objects[i], objects[j], *this);
                    }
                }
            }

        };
    }
};