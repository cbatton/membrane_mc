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

#ifndef OUTPUT_SYSTEM_
#define OUTPUT_SYSTEM_

class output_system {
    public:
        void outputTriangulation(string);
        void outputTriangulationAppend(string);
        void outputTriangulationStorage();
        void dumpXYZConfig(string);
        void dumpXYZConfigNormal(string);
        void dumpXYZCheckerboard(string);
        void dumpPhiNode(string);
        void dumpAreaNode(string);

};

#endif
