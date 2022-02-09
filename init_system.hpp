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
        InitSystem(MembraneMC*);
        ~InitSystem();
        void InitializeEquilState();
        void SaruSeed(unsigned int);
        void GenerateTriangulationEquil();
        inline int LinkTriangleTest(int, int);
        void UseTriangulation(string);
        void UseTriangulationRestart(string);

        // MembraneMC pointer
        MembraneMC* sys;
};

#endif
