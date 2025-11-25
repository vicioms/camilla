#pragma once
#include "../generic.h"
#include <vector>
#include <random>
#include <array>
using namespace std;
using tetraomino = std::array<int2,4>;


static const tetraomino straight_base = {{{0,0},{0,1},{0,2},{0,3}}};
static const tetraomino square_base = {{{0,0},{0,1},{1,1},{1,0}}};
static const tetraomino t_base = {{{0,0},{0,1},{0,2},{1,1}}};
static const tetraomino l_base = {{{0,0},{1,0},{2,0},{2,1}}};
static const tetraomino skew_base = {{{1,0},{1,1},{0,1},{0,2}}};



void tetraomino_translate(tetraomino& t, int2 v)
{
    for(int i=0; i<4; i++)
        t[i] = {t[i].x + v.x, t[i].y + v.y};
};
void tetraomino_translate_pbc(tetraomino& t, int2 v, int2 n)
{
    for(int i=0; i<4; i++)
    {
        t[i] = {pmod(t[i].x + v.x, n.x), pmod(t[i].y + v.y, n.y)};
    }
};
void tetraomino_translate_inv(tetraomino& t, int2 v)
{
    for(int i=0; i<4; i++)
        t[i] = {t[i].x - v.x, t[i].y - v.y};
};
void tetraomino_translate_inv_pbc(tetraomino& t, int2 v, int2 n)
{
    for(int i=0; i<4; i++)
    {
        t[i] = {pmod(t[i].x - v.x, n.x), pmod(t[i].y - v.y, n.y)};
    }
};

void tetraomino_rot90(tetraomino& t, int2 origin)
{
    tetraomino_translate_inv(t, origin);
    for(int i=0; i<4; i++)
        t[i] = {-t[i].y, t[i].x};
    tetraomino_translate(t, origin);
};
void tetraomino_rot180(tetraomino& t, int2 origin)
{
    tetraomino_translate_inv(t, origin);
    for(int i=0; i<4; i++)
        t[i] = {-t[i].x, -t[i].y};
    tetraomino_translate(t, origin);
};
void tetraomino_rot270(tetraomino& t, int2 origin)
{
    tetraomino_translate_inv(t, origin);
    for(int i=0; i<4; i++)
        t[i] = {t[i].x, -t[i].y};
    tetraomino_translate(t, origin);
};
void tetraomino_rot90(tetraomino& t, int origin)
{
    tetraomino_rot90(t, t[origin]);
};
void tetraomino_rot180(tetraomino& t, int origin)
{
    tetraomino_rot180(t, t[origin]);
};
void tetraomino_rot270(tetraomino& t, int origin)
{
    tetraomino_rot270(t, t[origin]);
};
bool is_move_allowed(const tetraomino& new_t, int id, int* board, int n_rows, int n_cols)
{
    for(int i=0; i<4; i++)
    {
        int2 p = new_t[i];
        if(p.x < 0 or p.x >= n_rows or p.y < 0 or p.y >= n_cols)
            return false;
        int p_id = board[pack(p.x, p.y, n_cols)];
        if(p_id == 0 or p_id == id)
            continue;
        else
            return false;
    }
    return true;
};
bool is_swap_allowed(const tetraomino& t1, const tetraomino& t2, int origin_1, int origin_2, int id1, int id2, int* board, int n_rows, int n_cols)
{
    return false;
};

void init_board(int n_rows, int n_cols, int*& board, vector<tetraomino>& particles, vector<int>& ids, default_random_engine& eng)
{
    uniform_int_distribution<int> type_distr(0, 4);
    board = (int*)malloc(n_rows * n_cols * sizeof(int));
    fill(board,n_rows * n_cols, 0);
    int last_id = 1;
    for(int i = 0; i < n_rows; i+= 4)
    {
        int end_i = i + 4;
        if(end_i > n_rows)
            continue;
        for(int j =0; j < n_cols; j+=4)
        {
            int end_j = j + 4;
            if(end_j > n_cols)
                continue;
            int type = type_distr(eng);
            tetraomino particle;
            switch (type)
            {
            case 0:
                particle = straight_base;
                break;
            case 1:
                particle = square_base;
                break;
            case 2:
                particle = t_base;
                break;
            case 3:
                particle = l_base;
                break;
            case 4:
                particle = skew_base;
                break;
            default:
                continue;
            }
            tetraomino_translate(particle, {i,j});
            particles.push_back(particle);
            ids.push_back(last_id);
            
            for(int k=0; k<4; k++)
            {
                int2 p = particle[k];
                board[pack(p.x, p.y, n_cols)] = last_id;
            };
            last_id += 1;
        }
    }
    
}