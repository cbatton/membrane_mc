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

#ifndef INIT_SYSTEM_
#define INIT_SYSTEM_

class InitSystem {
    public:
        void InitializeState();
        void InitializeEquilState();
        void InputState();
        void SaruSeed(unsigned int);
        void GenerateTriangulation();
        void GenerateTriangulationEquil();
        inline int LinkTriangleTest(int, int);
        inline void LinkTriangleFace(int, int, int *);
        void UseTriangulation(string);
        void UseTriangulationRestart(string);

};

#endif
