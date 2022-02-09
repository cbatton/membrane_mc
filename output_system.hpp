#ifndef OUTPUT_SYSTEM_
#define OUTPUT_SYSTEM_

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
using namespace std;

class OutputSystem {
    public:
        OutputSystem();
        ~OutputSystem();
        void OutputTriangulation(MembraneMC&, string);
        void OutputTriangulationAppend(MembraneMC&, string);
        void OutputTriangulationStorage(MembraneMC&);
        void DumpXYZConfig(MembraneMC&,string);
};

#endif
