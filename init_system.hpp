#ifndef INIT_SYSTEM_
#define INIT_SYSTEM_

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

class InitSystem {
    public:
        InitSystem();
        ~InitSystem();
        void Initialize(MembraneMC&);
        void InitializeEquilState(MembraneMC&);
        void GenerateTriangulationEquil(MembraneMC&);
        inline int LinkTriangleTest(MembraneMC&, int, int);
        void UseTriangulation(MembraneMC&, string);
        void UseTriangulationRestart(MembraneMC&, string);
};

#endif
