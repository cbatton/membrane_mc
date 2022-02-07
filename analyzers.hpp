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

class Analyzers {
    public:
        Analyzers();
        ~Analyzers();
        void EnergyAnalyzer();
        void AreaAnalyzer();
        void AreaProjAnalyzer();
        void MassAnalyzer();
        void NumberNeighborsAnalyzer();
        void UmbAnalyzer();
        void UmbOutput(int, ofstream&);
        void SampleNumberNeighbors(int);
        void DumpNumberNeighbors(string, int);
        void ClusterAnalysis();
        void ClusterPostAnalysis();
        void ClusterDFS(int, int, cluster&);
        void RDFRoutine(int, int, int, int, int);
        void RhoSample();
        void RhoAnalyzer();

        // MembraneMC pointer
        MembraneMC* sys;

        // Storage variables
        int storage_time = 10;
        int storage_neighbor =  10;
        int storage_umb_time = 100;
        vector<double> energy_storage;
        vector<double> area_storage;
        vector<double> area_proj_storage;
        vector<double> mass_storage;
        vector<vector<long long int>> numbers_neighbor;
        vector<double> energy_storage_umb;
        vector<double> energy_harmonic_umb;
        // Analyzer counts
        int storage_counts = 0;
        int umb_counts = 0;
        int neighbor_counts = 0;

        // Mean density from protein center variables
        int rdf_sample = 0;
        double mass_sample[3] = {0,0,0};
        int bins = 26;
        double bin_size = 10;
        vector<vector<double>> rho;

        // Cluster data type
        struct cluster {
            vector<int> vertex_status;
            vector<vector<int>> cluster_list;
        };

        vector<double> mean_cluster_number;
        vector<double> mean_cluster_weight;
        vector<double> mean_cluster_number_protein;
        vector<double> mean_cluster_weight_protein;
        vector<int> number_clusters;

};

#endif
