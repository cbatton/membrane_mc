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
using namespace std;

#ifndef MMC_H_
#define MMC_H_

class membrane_mc {

    // Going to organize these things to help keep things sane
    // Functions
    void initializeState();
    void initializeEquilState();
    void inputParam();
    void inputState();
    void SaruSeed(unsigned int);
    void generateTriangulation();
    void generateTriangulationEquil();
    inline int link_triangle_test(int, int);
    inline void link_triangle_face(int, int, int *);
    void useTriangulation(string);
    void useTriangulationRestart(string);
    void outputTriangulation(string);
    void outputTriangulationAppend(string);
    void outputTriangulationStorage();
    double wrapDistance_x(double, double);
    double wrapDistance_y(double, double);
    double lengthLink(int, int);
    void generateNeighborList();
    void areaNode(int);
    void normalTriangle(int i, double normal[3]);
    void shuffle_saru(Saru&, vector<int>&);
    void generateCheckerboard();
    double cosineAngle(int, int, int);
    double cosineAngle_norm(int, int, int);
    double cotangent(double);
    void acos_fast_initialize();
    inline double acos_fast(double);
    void cotangent_fast_initialize();
    inline double cotangent_fast(double);
    void linkMaxMin();
    void energyNode(int);
    void initializeEnergy();
    void initializeEnergy_scale();
    void energyNode_i(int);
    void initializeEnergy_i();
    void DisplaceStep(int = -1, int = 0);
    void TetherCut(int = -1, int = 0);
    void ChangeMassNonCon(int = -1, int = 0);
    void ChangeMassNonCon_gl(int = -1);
    void ChangeMassCon(int = -1, int = 0);
    void ChangeMassCon_nl();
    void moveProtein_gen(int, int);
    void moveProtein_nl(int, int, int);
    void ChangeArea();
    void CheckerboardMCSweep(bool);
    void nextStepSerial();
    void nextStepParallel(bool);
    void dumpXYZConfig(string);
    void dumpXYZConfigNormal(string);
    void dumpXYZCheckerboard(string);
    void dumpPhiNode(string);
    void dumpAreaNode(string);
    void sampleNumberNeighbors(int);
    void dumpNumberNeighbors(string, int);
    void equilibriate(int, chrono::steady_clock::time_point&);
    void simulate(int, chrono::steady_clock::time_point&);
    void energyAnalyzer();
    void areaAnalyzer();
    void areaProjAnalyzer();
    void massAnalyzer();
    void numberNeighborsAnalyzer();
    void umbAnalyzer();
    void umbOutput(int, ofstream&);

    // Variables
    // Initial mesh is points distributed in rectangular grid
    const int dim_x = 200; // Nodes in x direction
    const int dim_y = 200; // Nodes in y direction
    // Triangle properties
    const int vertices = dim_x*dim_y;
    const int faces = 2*dim_x*dim_y;
    int active_vertices = vertices;
    int active_faces = faces;

    // Note total number of nodes is dim_x*dim_y
    double Radius_x[dim_x][dim_y]; // Array to store x coordinate of node
    double Radius_y[dim_x][dim_y]; // Array to store y coordinate of node
    double Radius_z[dim_x][dim_y]; // Array to store z coordinate of node
    double Radius_x_original[dim_x][dim_y];
    double Radius_y_original[dim_x][dim_y];
    double Radius_z_original[dim_x][dim_y];
    // Triangulation radius values
    double Radius_x_tri[vertices];
    double Radius_y_tri[vertices];
    double Radius_z_tri[vertices];
    double Radius_x_tri_original[vertices];
    double Radius_y_tri_original[vertices];
    double Radius_z_tri_original[vertices];
    int Ising_Array[vertices];
    // Triangles
    const int neighbor_min = 2;
    const int neighbor_max = 10;
    int triangle_list[faces][3];
    int point_neighbor_max[vertices];
    int point_neighbor_list[vertices][neighbor_max];
    int point_triangle_list[vertices][neighbor_max];
    int point_triangle_max[vertices];
    int point_neighbor_triangle[vertices][neighbor_max][2];

    int triangle_list_original[faces][3];
    int point_neighbor_list_original[vertices][neighbor_max];
    int point_neighbor_max_original[vertices];
    int point_triangle_list_original[vertices][neighbor_max];
    int point_triangle_max_original[vertices];
    int point_neighbor_triangle_original[vertices][neighbor_max][2];

    // Neighborlist
    vector<vector<int>> neighbor_list; // Neighbor list
    vector<int> neighbor_list_index; // Map from particle to index
    vector<vector<int>> neighbors; // Neighboring bins
    vector<int> index_particles_max_nl;
    // Indexing for neighbor list
    int nl_x = int(9000*2.0)/1.00-1;
    int nl_y = int(9000*2.0)/1.00-1;
    int nl_z = int(60*2.0)/1.00-1;

    // Checkerboard set
    vector<vector<int>> checkerboard_list; // Neighbor list
    vector<int> checkerboard_index; // Map from particle to index
    vector<vector<int>> checkerboard_neighbors; // Neighboring bins
    // Indexing for neighbor list
    int checkerboard_x = 1;
    int checkerboard_y = 1;
    double checkerboard_set_size = 3.5;

    double phi_vertex[vertices];
    double phi_vertex_original[vertices];
    double mean_curvature_vertex[vertices];
    double mean_curvature_vertex_original[vertices];
    double sigma_vertex[vertices];
    double sigma_vertex_original[vertices];
    double Phi = 0.0; // Energy at current step
    double Phi_phi = 0.0; // Composition energy at current step
    double Phi_bending = 0.0; // Bending energy at current step
    int Mass = 0;
    double Magnet = 0;
    double area_faces[faces];
    double area_faces_original[faces];
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
    double box_x = Length_x/double(nl_x); 
    double box_y = Length_y/double(nl_y); 
    double box_z = 2*Length_z/double(nl_z);
    double box_x_checkerboard = Length_x/checkerboard_x;
    double box_y_checkerboard = Length_y/checkerboard_y;
    double cell_center_x = 0.0;
    double cell_center_y = 0.0;
    double lambda = 0.0075; // Maximum percent change in displacement
    double lambda_scale = 0.01; // Maximum percent change in scale
    double num_frac = 0.5; // Number fraction of types
    double scale = 1.0;
    const int Cycles_eq = 1000001;
    const int Cycles_prod = 1000001;
    double T = 2.0; // Temperature
    double T_list[2] = {2.0, 2.0};
    long long int steps_tested_displace = 0;
    long long int steps_rejected_displace = 0;
    long long int steps_tested_tether = 0;
    long long int steps_rejected_tether = 0;
    long long int steps_tested_mass = 0;
    long long int steps_rejected_mass = 0;
    long long int steps_tested_protein = 0;
    long long int steps_rejected_protein = 0;
    long long int steps_tested_area = 0;
    long long int steps_rejected_area = 0;
    long long int steps_tested_eq = 0;
    long long int steps_rejected_eq = 0;
    long long int steps_tested_prod = 0;
    long long int steps_rejected_prod = 0;
    // nl move parameter
    int nl_move_start = 0;

    // Protein variables
    int protein_node[vertices];
    int num_proteins = 2;

    // Storage variables
    const int storage_time = 10;
    const int storage = Cycles_prod/storage_time;
    int storage_counts = 0;
    double energy_storage[storage+1];
    double area_storage[storage+1];
    double area_proj_storage[storage+1];
    double mass_storage[storage+1];
    // Analyzer
    const int storage_neighbor =  10;
    const int storage_umb_time = 100;
    const int storage_umb = Cycles_prod/storage_umb_time;
    int umb_counts = 0;
    int neighbor_counts = 0;
    long long int numbers_neighbor[neighbor_max][Cycles_prod/storage_neighbor];
    double energy_storage_umb[storage_umb];
    double energy_harmonic_umb[storage_umb];

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

    // Look up table variables
    const int look_up_density = 10001;
    double acos_lu_min = -1.0;
    double acos_lu_max = 1.0;
    double cot_lu_min = 0.15;
    double cot_lu_max = M_PI-0.15;
    double lu_interval_acos = (acos_lu_max-acos_lu_min)/(look_up_density-1);
    double lu_interval_cot = (cot_lu_max-cot_lu_min)/(look_up_density-1);
    double in_lu_interval_acos = 1.0/lu_interval_acos;
    double in_lu_interval_cot = 1.0/lu_interval_cot;

    // Arrays to store look up tables
    double lu_points_acos[look_up_density];
    double lu_values_acos[look_up_density];
    double lu_points_cot[look_up_density];
    double lu_values_cot[look_up_density];

    // MPI variables
    int world_size=1;
    int world_rank=0;
    string local_path;
    vector<string> rank_paths;
    string output;
    string output_path;
    streambuf *myfilebuf;

    // Testing
    double max_diff = -1;
    double relative_diff = 0;

    // openmp stuff
    const int max_threads = 272;
    int active_threads = 0;
    double Phi_diff_thread[max_threads][8];
    double Phi_phi_diff_thread[max_threads][8];
    double Phi_bending_diff_thread[max_threads][8];
    double Area_diff_thread[max_threads][8];
    int Mass_diff_thread[max_threads][8];
    double Magnet_diff_thread[max_threads][8];
    int steps_tested_displace_thread[max_threads][8];
    int steps_rejected_displace_thread[max_threads][8];
    int steps_tested_tether_thread[max_threads][8];
    int steps_rejected_tether_thread[max_threads][8];
    int steps_tested_mass_thread[max_threads][8];
    int steps_rejected_mass_thread[max_threads][8];
    int steps_tested_protein_thread[max_threads][8];
    int steps_rejected_protein_thread[max_threads][8];
    /*
    int omp_get_max_threads();
    int omp_get_thread_num();
    int omp_get_num_threads();
    */
};

#endif
