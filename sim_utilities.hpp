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

#ifndef SIM_UTILITIES_
#define SIM_UTILITIES_

class SimUtilities {
    public:
        SimUtilities(MembraneMC*);
        ~SimUtilities();
        double WrapDistance(double, double);
        double LengthLink(int, int);
        void AreaNode(int);
        void NormalTriangle(int i, double normal[3]);
        void ShuffleSaru(Saru&, vector<int>&);
        double Cotangent(int, int, int);

        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
