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
#include "neighborlist.hpp"
#include "saruprng.hpp"
using namespace std;

class Utilities {
    public:
        Utilities();
        ~Utilities();
        void LinkMaxMin(MembraneMC&, NeighborList&);
        void EnergyNode(MembraneMC&, int);
        void InitializeEnergy(MembraneMC&, NeighborList&);
        void InitializeEnergyScale(MembraneMC&, NeighborList&);
        double WrapDistance(double, double);
        double LengthLink(MembraneMC&, int, int);
        void AreaNode(MembraneMC&, int);
        void NormalTriangle(MembraneMC&, int i, double normal[3]);
        void ShuffleSaru(Saru&, vector<int>&);
        double Cotangent(MembraneMC&, int, int, int);
};

#endif
