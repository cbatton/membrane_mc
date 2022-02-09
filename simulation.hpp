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

class Simulation {
    public:
        Simulation(MembraneMC*);
        ~Simulation();
        void CheckerboardMCSweep(bool);
        void NextStepSerial();
        void NextStepParallel(bool);

        // openmp stuff
        const int max_threads = omp_get_max_threads();
        int active_threads = 0;
        double phi_diff_thread[max_threads][8];
        double phi_phi_diff_thread[max_threads][8];
        double phi_bending_diff_thread[max_threads][8];
        double area_diff_thread[max_threads][8];
        int mass_diff_thread[max_threads][8];
        double magnet_diff_thread[max_threads][8];

        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
