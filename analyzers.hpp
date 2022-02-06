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

#ifndef ANALYZERS_
#define ANALYZERS_

class analyzers {
    public:
        void energyAnalyzer();
        void areaAnalyzer();
        void areaProjAnalyzer();
        void massAnalyzer();
        void numberNeighborsAnalyzer();
        void umbAnalyzer();
        void umbOutput(int, ofstream&);
        void sampleNumberNeighbors(int);
        void dumpNumberNeighbors(string, int);

        // Storage variables
        int storage_time = 10;
        int storage = Cycles_prod/storage_time;
        vector<double> energy_storage;
        vector<double> area_storage;
        vector<double> area_proj_storage;
        vector<double> mass_storage;
        // Analyzer
        int storage_neighbor =  10;
        int storage_umb_time = 100;
        int storage_umb = Cycles_prod/storage_umb_time;
        int umb_counts = 0;
        int neighbor_counts = 0;
        vector<vector<long long int>> numbers_neighbor;
        vector<double> energy_storage_umb;
        vector<double> energy_harmonic_umb;


};

#endif
