#include "polyom.h"
#include <fstream>
#include <iostream>
using namespace std;



int main()
{
    int n_rows = 128;
    int n_cols = 128;
    int* board;
    vector<tetraomino> particles;
    vector<int> ids;
    default_random_engine eng;
    init_board(n_rows, n_cols, board, particles, ids, eng);
    //printf("Initialized board with %d particles\n", (int)particles.size());
    uniform_int_distribution<int> mc_particle_selector(0, particles.size()-1);
    ofstream fout("board.dat");
    for(int step=0; step<500000; step++)
    {
        if(step % 5000 == 0)
            fout.write((char*)board, n_rows*n_cols*sizeof(int));
        for(int move = 0; move < particles.size(); move++)
        {
            int p_index = mc_particle_selector(eng);
            tetraomino& t = particles[p_index];
            int id = ids[p_index];
            tetraomino new_t = t;
            uniform_int_distribution<int> move_type_distr(0,6);
            int move_type = move_type_distr(eng);
            switch(move_type)
            {
                case 0:
                    tetraomino_translate(new_t, {1,0});
                    break;
                case 1:
                    tetraomino_translate(new_t, {-1,0});
                    break;
                case 2:
                    tetraomino_translate(new_t, {0,1});
                    break;
                case 3:
                    tetraomino_translate(new_t, {0,-1});
                    break;
                case 4:
                    tetraomino_rot90(new_t, 0);
                    break;
                case 5:
                    tetraomino_rot180(new_t, 0);
                    break;
                case 6:
                    tetraomino_rot270(new_t, 0);
                    break;
                default:
                    continue;
            }
            if(is_move_allowed(new_t, id, board, n_rows, n_cols))
            {
                // clear old position
                for(int i=0; i<4; i++)
                {
                    int2 p = t[i];
                    board[pack(p.x, p.y, n_cols)] = 0;
                }
                // set new position
                for(int i=0; i<4; i++)
                {
                    int2 p = new_t[i];
                    board[pack(p.x, p.y, n_cols)] = id;
                }
                t = new_t;
            }
        }
    }; 
    fout.close();
    free(board);
    return 0;
}


