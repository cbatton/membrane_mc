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

#ifndef SIMULATION_
#define SIMULATION_

class simulation {
    public:
        void CheckerboardMCSweep(bool);
        void nextStepSerial();
        void nextStepParallel(bool);
        void equilibriate(int, chrono::steady_clock::time_point&);
        void simulate(int, chrono::steady_clock::time_point&);

        int Cycles_eq = 1000001;
        int Cycles_prod = 1000001;

        // openmp stuff
        const int max_threads = omp_get_max_threads();
        int active_threads = 0;
        double Phi_diff_thread[max_threads][8];
        double Phi_phi_diff_thread[max_threads][8];
        double Phi_bending_diff_thread[max_threads][8];
        double Area_diff_thread[max_threads][8];
        int Mass_diff_thread[max_threads][8];
        double Magnet_diff_thread[max_threads][8];
        int steps_tested_displace_thread[max_threads][8];
        int steps_rejected_displace_thread[max_threads][8];
        int steps_tested_tether_thread[max_threads][8];
        int steps_rejected_tether_thread[max_threads][8];
        int steps_tested_mass_thread[max_threads][8];
        int steps_rejected_mass_thread[max_threads][8];
        int steps_tested_protein_thread[max_threads][8];
        int steps_rejected_protein_thread[max_threads][8];
};

#endif
