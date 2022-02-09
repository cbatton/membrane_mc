#ifndef UTILITIES_
#define UTILITIES_

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

class Utilities {
    public:
        Utilities(MembraneMC*);
        ~Utilities();
        void LinkMaxMin();
        void EnergyNode(int);
        void InitializeEnergy();
        void InitializeEnergyScale();

        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
