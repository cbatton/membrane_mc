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

#ifndef UTILITIES_
#define UTILITIES_

class Utilities {
    public:
        void LinkMaxMin();
        void EnergyNode(int);
        void InitializeEnergy();
        void InitializeEnergyScale();
        void EnergyNode_i(int);
        void InitializeEnergy_i();

};

#endif
