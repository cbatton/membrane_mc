#ifndef NEIGHBORLIST_
#define NEIGHBORLIST_

#include <iostream> 
#include <fstream>
#include <random>
#include <algorithm>
#include <math.h>
#include <ctime>
#include <string>
#include <vector>
#include "omp.h"
#include <mpi.h>
#include <iomanip>
#include <chrono>
#include "membrane_mc.hpp"
#include "saruprng.hpp"
using namespace std;

class NeighborList {
    public:
        NeighborList(MembraneMC*);
        ~NeighborList();
        void GenerateNeighborList();
        void GenerateCheckerboard();

        // Neighborlist
        vector<vector<int>> neighbor_list; // Neighbor list
        vector<int> neighbor_list_index; // Map from particle to index
        vector<vector<int>> neighbors; // Neighboring bins
        vector<int> index_particles_max_nl;
        // Indexing for neighbor list
        int nl_x = int(9000*2.0)/1.00-1;
        int nl_y = int(9000*2.0)/1.00-1;
        int nl_z = int(60*2.0)/1.00-1;

        // Checkerboard set
        vector<vector<int>> checkerboard_list; // Neighbor list
        vector<int> checkerboard_index; // Map from particle to index
        vector<vector<int>> checkerboard_neighbors; // Neighboring bins
        // Indexing for neighbor list
        int checkerboard_x = 1;
        int checkerboard_y = 1;
        double checkerboard_set_size = 3.5;
        double box_x = 9000.0/double(nl_x); 
        double box_y = 9000.0/double(nl_y); 
        double box_z = 2*9000.0/double(nl_z);
        double box_x_checkerboard = 9000.0/checkerboard_x;
        double box_y_checkerboard = 9000.0/checkerboard_y;
        double cell_center_x = 0.0;
        double cell_center_y = 0.0;

        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
