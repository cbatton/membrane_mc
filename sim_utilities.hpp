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
        double WrapDistance(double, double, double);
        double LengthLink(int, int);
        void AreaNode(int);
        void NormalTriangle(int i, double normal[3]);
        void ShuffleSaru(Saru&, vector<int>&);

        // Want to use one where I compute cotangents directly using cos, sin
        double CosineAngle(int, int, int); // Will use a different scheme here.....
        double CosineAngleNorm(int, int, int);
        double Cotangent(double);
        double Cotangent(int, int, int);
        void AcosFastInitialize();
        inline double AcosFast(double);
        void CotangentFastInitialize();
        inline double CotangentFast(double);

        // Look up table variables
        const int look_up_density = 10001;
        double acos_lu_min = -1.0;
        double acos_lu_max = 1.0;
        double cot_lu_min = 0.15;
        double cot_lu_max = M_PI-0.15;
        double lu_interval_acos = (acos_lu_max-acos_lu_min)/(look_up_density-1);
        double lu_interval_cot = (cot_lu_max-cot_lu_min)/(look_up_density-1);
        double in_lu_interval_acos = 1.0/lu_interval_acos;
        double in_lu_interval_cot = 1.0/lu_interval_cot;

        // Arrays to store look up tables
        double lu_points_acos[look_up_density];
        double lu_values_acos[look_up_density];
        double lu_points_cot[look_up_density];
        double lu_values_cot[look_up_density];

};

#endif
