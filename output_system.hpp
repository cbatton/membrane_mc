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

class OutputSystem {
    public:
        OutputSystem(MembraneMC*);
        ~OutputSystem();
        void OutputTriangulation(string);
        void OutputTriangulationAppend(string);
        void OutputTriangulationStorage();
        void DumpXYZConfig(string);

        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
