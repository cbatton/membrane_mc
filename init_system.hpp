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

class init_system {
    public:
        void initializeState();
        void initializeEquilState();
        void inputState();
        void SaruSeed(unsigned int);
        void generateTriangulation();
        void generateTriangulationEquil();
        inline int link_triangle_test(int, int);
        inline void link_triangle_face(int, int, int *);
        void useTriangulation(string);
        void useTriangulationRestart(string);

};

#endif
