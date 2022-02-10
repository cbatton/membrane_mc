#ifndef SIMULATION_
#define SIMULATION_

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
#include "analyzers.hpp"
#include "saruprng.hpp"
#include "output_system.hpp"
#include "mc_moves.hpp"
#include "utilities.hpp"
using namespace std;

class Simulation {
    public:
        Simulation(double, double, int, int);
        ~Simulation();
        void CheckerboardMCSweep(bool, MembraneMC&, NeighborList&);
        void NextStepSerial(MembraneMC&, NeighborList&);
        void NextStepParallel(bool, MembraneMC&, NeighborList&);
        void Equilibriate(int, MembraneMC&, NeighborList&, chrono::steady_clock::time_point&);
        void Simulate(int, MembraneMC&, NeighborList&, Analyzers&, chrono::steady_clock::time_point&);
        // Helper classes
        MCMoves mc_mover;
        OutputSystem output;
        Utilities util;
};

#endif
