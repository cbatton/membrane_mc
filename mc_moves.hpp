#ifndef MC_MOVES_
#define MC_MOVES_

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
#include "neighborlist.hpp"
#include "utilities.hpp"
#include "saruprng.hpp"
using namespace std;


class MCMoves {
    public:
        MCMoves(double, double, int, int);
        ~MCMoves();
        void DisplaceStep(MembraneMC&, NeighborList&, int, int);
        void TetherCut(MembraneMC&, NeighborList&, int, int);
        void ChangeMassNonCon(MembraneMC&, NeighborList&, int, int);
        void ChangeMassCon(MembraneMC&, NeighborList&, int, int);
        void MoveProteinGen(MembraneMC&, NeighborList&, int, int);
        void MoveProteinNL(MembraneMC&, int, int, int);
        void ChangeArea(MembraneMC&, NeighborList&);

        double lambda = 0.0075; // Maximum percent change in displacement
        double lambda_scale = 0.01; // Maximum percent change in scale
        long long int steps_tested_displace = 0;
        long long int steps_rejected_displace = 0;
        long long int steps_tested_tether = 0;
        long long int steps_rejected_tether = 0;
        long long int steps_tested_mass = 0;
        long long int steps_rejected_mass = 0;
        long long int steps_tested_protein = 0;
        long long int steps_rejected_protein = 0;
        long long int steps_tested_area = 0;
        long long int steps_rejected_area = 0;
        long long int steps_tested_eq = 0;
        long long int steps_rejected_eq = 0;
        long long int steps_tested_prod = 0;
        long long int steps_rejected_prod = 0;
        // nl move parameter
        int nl_move_start = 0;
        int max_threads = 272;
        //if(max_threads != omp_get_max_threads()){
            //max_threads = omp_get_max_threads();
        //}
        vector<vector<int>> steps_tested_displace_thread;
        vector<vector<int>> steps_rejected_displace_thread;
        vector<vector<int>> steps_tested_tether_thread;
        vector<vector<int>> steps_rejected_tether_thread;
        vector<vector<int>> steps_tested_mass_thread;
        vector<vector<int>> steps_rejected_mass_thread;
        vector<vector<int>> steps_tested_protein_thread;
        vector<vector<int>> steps_rejected_protein_thread;
        // Utilities helper
        Utilities util;
};

#endif
