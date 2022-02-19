#ifndef ANALYZERS_
#define ANALYZERS_

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

struct cluster {
    vector<int> vertex_status;
    vector<vector<int>> cluster_list;
};


class Analyzers {
    public:
        Analyzers();
        Analyzers(int, int, int, MembraneMC&); // constructor to call after more info is there
        ~Analyzers();
        void EnergyAnalyzer();
        void AreaAnalyzer();
        void AreaProjAnalyzer();
        void MassAnalyzer();
        void UmbOutput(double&, double&, double&, vector<double>&, double&, ofstream&);
        void ClusterAnalysis(MembraneMC&);
        void ClusterPostAnalysis();
        void ClusterDFS(MembraneMC&, int, int, cluster&);
        void RDFRoutine(MembraneMC&, int, int, int, int, int);
        void RhoSample(MembraneMC&);
        void RhoAnalyzer(MembraneMC&);
        void OutputAnalyzers(MembraneMC&);

        // Mean density from protein center variables
        double area_proj_average;
        int rdf_sample = 0;
        double mass_sample[3] = {0,0,0};
        int bins;
        double bin_size = 10;
        vector<vector<double>> rho;

        // Storage variables
        int storage_time;
        int storage_umb_time;
        vector<double> energy_storage;
        vector<double> area_storage;
        vector<double> area_proj_storage;
        vector<double> mass_storage;
        vector<vector<long long int>> numbers_neighbor;
        // Analyzer counts
        int storage_counts = 0;
        int umb_counts = 0;
        int neighbor_counts = 0;
        // Output string
        string output_path;

        vector<double> mean_cluster_number;
        vector<double> mean_cluster_weight;
        vector<double> mean_cluster_number_protein;
        vector<double> mean_cluster_weight_protein;
        vector<int> number_clusters;

};

#endif
