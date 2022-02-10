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
using namespace std;

class Simulation {
    public:
        Simulation();
        ~Simulation();
        void CheckerboardMCSweep(bool);
        void NextStepSerial();
        void NextStepParallel(bool);
        void Equilibriate(int, MembraneMC&, chrono::steady_clock::time_point&);
        void Simulate(int, MembraneMC&, Analyzers&, chrono::steady_clock::time_point&);
};

#endif
