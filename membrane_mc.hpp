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
#include "saruprng.hpp"
#include "analyzers.hpp"
#include "init_system.hpp"
#include "mc_moves.hpp"
#include "neighborlist.hpp"
#include "output_system.hpp"
#include "simulation.hpp"
#include "sim_utilities.hpp"
#include "utilities.hpp"
using namespace std;

#ifndef MMC_H_
#define MMC_H_

class MembraneMC {
    public:
        // This class contains all the major generic functions I am going to call
        // Functions
        void InputParam();
        void Equilibriate(int, chrono::steady_clock::time_point&);
        void Simulate(int, chrono::steady_clock::time_point&);
        void OutputTimes();
        void OutputAnalyzers();
        // Variables
        // Class variables
        shared_ptr<analyzers> analysis;
        shared_ptr<init_system> initializer;
        shared_ptr<mc_moves> mc_mover;
        shared_ptr<neighborlist> nl;
        shared_ptr<output_system> output;
        shared_ptr<simulation> sim;
        shared_ptr<sim_utilities> sim_util;
        shared_ptr<utilities> util;
        // Variables
        // Initial mesh is points distributed in rectangular grid
        int dim_x = 200; // Nodes in x direction
        int dim_y = 200; // Nodes in y direction
        // Triangle properties
        int vertices = dim_x*dim_y;
        int faces = 2*dim_x*dim_y;
        int active_vertices = vertices;
        int active_faces = faces;

        // Triangulation radius values
        vector<vector<double> Radius_tri;
        vector<vector<double> Radius_tri_original;
        vector<int> Ising_Array;
        // Triangles
        const int neighbor_min = 2;
        const int neighbor_max = 10;

        vector<vector<int>> triangle_list;
        vector<vector<int>> point_neighbor_list;
        vector<vector<int>> point_triangle_list;
        vector<vector<vector<int>>> point_neighbor_triangle;

        vector<vector<int>> triangle_list_original;
        vector<vector<int>> point_neighbor_list_original;
        vector<vector<int>> point_triangle_list_original;
        vector<vector<vector<int>>> point_neighbor_triangle_original;


        vector<double> phi_vertex;
        vector<double> phi_vertex_original;
        vector<double> mean_curvature_vertex;
        vector<double> mean_curvature_vertex_original;
        vector<double> sigma_vertex;
        vector<double> sigma_vertex_original;
        double Phi = 0.0; // Energy at current step
        double Phi_phi = 0.0; // Composition energy at current step
        double Phi_bending = 0.0; // Bending energy at current step
        int Mass = 0;
        double Magnet = 0;
        vector<double> area_faces;
        vector<double> area_faces_original;
        double Area_total;
        double sigma_i_total = 0.0;
        double Area_proj_average;

        double k_b[3] = {20.0, 20.0, 20.0}; // k units
        double k_g[3] = {0.0, 0.0, 0.0}; // k units
        double k_a[3] = {0.0, 0.0, 0.0}; // k units
        double gamma_surf[3] = {0.0, 0.0, 0.0}; // k units
        double tau_frame = 0;
        double ising_values[3] = {-1.0, 1.0, 1.0};
        double spon_curv[3] = {0.0,0.0,0.0}; // Spontaneous curvature at protein nodes
        double spon_curv_end = 0.0;
        double J_coupling[3][3] = {{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}};
        double h_external = 0.0;
        // double area_original_total = 0;
        // double area_current_total = 0;
        double Length_x = 96; // Length of a direction
        double Length_y = 96;
        double Length_z = 60;
        double Length_x_old = Length_x;
        double Length_y_old = Length_y;
        double Length_z_old = Length_z;
        double scale_xy = 1.0;
        double Length_x_base = Length_x;
        double Length_y_base = Length_y;
        double scale_xy_old = scale_xy;
        double num_frac = 0.5; // Number fraction of types
        double scale = 1.0;
        double T = 2.0; // Temperature
        double T_list[2] = {2.0, 2.0};

        // Protein variables
        vector<int> protein_node;
        int num_proteins = 2;

        // Pseudo-random number generators
        vector<Saru> generators;
        Saru generator;
        unsigned int seed_base = 0;
        unsigned int count_step = 0;

        // Time
        chrono::steady_clock::time_point t1;
        chrono::steady_clock::time_point t2;
        double final_time = 120.0;
        double final_warning = 60.0;
        double time_storage_cycle[2] = {0,0};
        double time_storage_area[4] = {0,0,0,0};
        double time_storage_displace[3] = {0,0,0};
        double time_storage_other[8] = {0,0,0,0,0,0,0,0};
        double time_storage_overall;

        // MPI variables
        int world_size=1;
        int world_rank=0;
        string local_path;
        vector<string> rank_paths;
        string output;
        string output_path;
        streambuf *myfilebuf;
        ofstream my_cout;
};

#endif
