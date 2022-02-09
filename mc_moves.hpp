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
#include "saruprng.hpp"
using namespace std;


class MCMoves {
    public:
        MCMoves(MembraneMC*);
        ~MCMoves();
        void DisplaceStep(int = -1, int = 0);
        void TetherCut(int = -1, int = 0);
        void ChangeMassNonCon(int = -1, int = 0);
        void ChangeMassCon(int = -1, int = 0);
        void MoveProteinGen(int, int);
        void MoveProteinNL(int, int, int);
        void ChangeArea();

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
        constexpr static int max_threads = 272;
        int steps_tested_displace_thread[max_threads][8];
        int steps_rejected_displace_thread[max_threads][8];
        int steps_tested_tether_thread[max_threads][8];
        int steps_rejected_tether_thread[max_threads][8];
        int steps_tested_mass_thread[max_threads][8];
        int steps_rejected_mass_thread[max_threads][8];
        int steps_tested_protein_thread[max_threads][8];
        int steps_rejected_protein_thread[max_threads][8];
        // nl move parameter
        int nl_move_start = 0;
        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
