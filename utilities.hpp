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

class utilities {
    public:
        void linkMaxMin();
        void energyNode(int);
        void initializeEnergy();
        void initializeEnergy_scale();
        void energyNode_i(int);
        void initializeEnergy_i();

};

#endif
