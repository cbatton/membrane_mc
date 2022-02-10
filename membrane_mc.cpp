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
#include <memory>
#include "membrane_mc.hpp"
#include "saruprng.hpp"
#include "analyzers.hpp"
#include "init_system.hpp"
#include "mc_moves.hpp"
#include "neighborlist.hpp"
#include "output_system.hpp"
#include "simulation.hpp"
#include "utilities.hpp"
using namespace std;

MembraneMC::MembraneMC(int active_threads) : active_threads(active_threads) {
    // Constructor
    // Initialize vector sizes
    phi_diff_thread.resize(active_threads,vector<double>(8,0));
    phi_phi_diff_thread.resize(active_threads,vector<double>(8,0));
    phi_bending_diff_thread.resize(active_threads,vector<double>(8,0));
    area_diff_thread.resize(active_threads,vector<double>(8,0));
    mass_diff_thread.resize(active_threads,vector<int>(8,0));
    magnet_diff_thread.resize(active_threads,vector<double>(8,0));
}

MembraneMC::~MembraneMC() {
    // Destructor
    // Does nothing
}

void MembraneMC::InputParam(int& argc, char* argv[]) { // Takes parameters from a file named param
    // MPI housekeeping
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // Get local path
    string path_file(argv[1]);
    ifstream input_0;
    input_0.open(path_file);
    input_0 >> local_path;
    input_0.close();
    // Get path for each processor
    string path_files(argv[2]);
    rank_paths.resize(world_size);
    ifstream input_1;
    string line;
    input_1.open(path_files);
    for(int i=0; i<world_size; i++) {
        input_1 >> rank_paths[i];
        getline(input_1,line);
    }
    input_1.close();
    // Set up output
    string output = local_path+rank_paths[world_rank]+"/out";
    output_path = local_path+rank_paths[world_rank];
    my_cout.open(output, ios_base::app);

    ifstream input;
    input.open(output_path+"/param");
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        if(world_rank == 0) {
            cout << "No input file." << endl;
        }
    }
    else{
        // Temporary variables to pass to constructors
        int bins = 26;
        int storage_time = 10;
        int storage_umb_time = 100;
        // Read in everything
        input >> line >> dim_x >> dim_y;
        getline(input,line);
        input >> line >> cycles_eq >> cycles_prod;
        getline(input, line);
        input >> line >> storage_time >> storage_umb_time;
        getline(input, line);
        input >> line >> lengths[0] >> lengths[1] >> lengths[2];
        getline(input,line);
        input >> line >> temp_list[0] >> temp_list[1];
        getline(input, line);
        input >> line >> k_b[0] >> k_b[1] >> k_b[2];
        getline(input, line);
        input >> line >> gamma_surf[0] >> gamma_surf[1] >> gamma_surf[2];
        getline(input, line);
        input >> line >> tau_frame;
        getline(input, line);
        input >> line >> spon_curv[0] >> spon_curv[1] >> spon_curv[2]; 
        getline(input, line);
        input >> line >> lambda;
        getline(input, line);
        input >> line >> lambda_scale;
        getline(input, line);
        input >> line >> ising_values[0] >> ising_values[1] >> ising_values[2];
        getline(input, line);
        input >> line >> j_coupling[0][0] >> j_coupling[0][1] >> j_coupling[0][2]; 
        getline(input, line);
        input >> j_coupling[1][0] >> j_coupling[1][1] >> j_coupling[1][2];
        getline(input, line);
        input >> j_coupling[2][0] >> j_coupling[2][1] >> j_coupling[2][2];
        getline(input, line);
        input >> line >> h_external;
        getline(input, line);
        input >> line >> num_frac;
        getline(input, line);
        input >> line >> num_proteins;
        getline(input, line);
        input >> line >> seed_base;
        getline(input, line);
        input >> line >> count_step;
        getline(input, line);
        input >> line >> final_time;
        getline(input, line);
        input >> line >> nl_move_start;
        getline(input, line);
        input >> line >> bins;
        input.close();
        // Output variables for catching things
        if(world_rank == 0) {
            cout << "Param file detected. Changing values." << endl;
            cout << "Dimensions is now " << dim_x << " " << dim_y << endl;
            cout << "Cycles is now " << cycles_eq << " " << cycles_prod << endl;
            cout << "Temperature is now " << temp_list[0] << " " << temp_list[1] << endl;
            cout << "k_b is now " << k_b[0] << " " << k_b[1] << " " << k_b[2] << endl;
            cout << "gamma_surf is now " << gamma_surf[0] << " " << gamma_surf[1] << " " << gamma_surf[2] << endl;
            cout << "tau_frame is now " << tau_frame << endl;
            cout << "Spontaneous curvature is now " << spon_curv[0] << " " << spon_curv[1] << " " << spon_curv[2] << endl;
            cout << "Lengths is now " << lengths[0] << " " << lengths[1] << " " << lengths[2] << endl;
            cout << "lambda is now " << lambda << endl;
            cout << "lambda_scale is now " << lambda_scale << endl;
            cout << "ising_values is now " << ising_values[0] << " " << ising_values[1] << " " << ising_values[2] << endl;
            cout << "J is now " << j_coupling[0][0] << " " << j_coupling[0][1] << " " << j_coupling[0][2] << endl;
            cout << "\t" << j_coupling[1][0] << " " << j_coupling[1][1] << " " << j_coupling[1][2] << endl;
            cout << "\t" << j_coupling[2][0] << " " << j_coupling[2][1] << " " << j_coupling[2][2] << endl;
            cout << "h is now " << h_external << endl;
            cout << "num_frac is now " << num_frac << endl;
            cout << "num_proteins is now " << num_proteins << endl;
            cout << "seed_base is now " << seed_base << endl;
            cout << "count_step is now " << count_step << endl;
            cout << "final_time is now " << final_time << endl;
            cout << "nl_move_start is now " << nl_move_start << endl;
            cout << "bins is now " << bins << endl;
        }
        // Set lengths and seed_base
        lengths_old = lengths;
        lengths_base = lengths;
        // Hash seed_base
        seed_base = seed_base*0x12345677 + 0x12345;
        seed_base = seed_base^(seed_base>>16);
        seed_base = seed_base*0x45679;
        if(world_rank == 0) {
            cout << "seed_base is now " << seed_base << endl;
        }
        // Initialize random number generators
        for(int i=0; i<omp_get_max_threads(); i++) {
            generators.push_back(Saru());
        }
        final_warning = final_time-60.0;
        spon_curv_end = spon_curv[2];
        spon_curv[2] = 0.0;
        // Initialize things
        vertices = dim_x*dim_y;
        faces = 2*vertices;
        // Particle variables
        radii_tri.resize(vertices,vector<double>(3,0.0));
        radii_tri_original.resize(vertices,vector<double>(3,0.0));
        ising_array.resize(vertices,0);
        phi_vertex.resize(vertices,0.0);
        phi_vertex_original.resize(vertices,0.0);
        mean_curvature_vertex.resize(vertices,0.0);
        mean_curvature_vertex_original.resize(vertices,0.0);
        sigma_vertex.resize(vertices,0.0);
        sigma_vertex_original.resize(vertices,0.0);
        area_faces.resize(faces,0.0);
        area_faces_original.resize(faces,0.0);
        // Triangle variables
        vector<int> list_int;
        triangle_list.resize(faces,list_int);
        triangle_list_original.resize(faces,list_int);
        for(int i=0; i<faces; i++) {
            triangle_list[i].resize(3,0);
            triangle_list_original[i].resize(3,0);
        }
        point_neighbor_list.resize(vertices,list_int);
        point_neighbor_list_original.resize(vertices,list_int);
        point_triangle_list.resize(vertices,list_int);
        point_triangle_list_original.resize(vertices,list_int);
        vector<vector<int>> list_int_int;
        point_neighbor_triangle.resize(vertices,list_int_int);
        point_neighbor_triangle_original.resize(vertices,list_int_int);
        for(int i=0; i<vertices; i++) {
            point_neighbor_list[i].resize(neighbor_max,-1);
            point_neighbor_list_original[i].resize(neighbor_max,-1);
            point_triangle_list[i].resize(neighbor_max,-1);
            point_triangle_list_original[i].resize(neighbor_max,-1);
            point_neighbor_triangle[i].resize(neighbor_max,list_int);
            point_neighbor_triangle_original[i].resize(neighbor_max,list_int);
            for(int j=0; j<neighbor_max; j++) {
                point_neighbor_triangle[i][j].resize(2,-1);
                point_neighbor_triangle_original[i][j].resize(2,-1);
            }
        }
        protein_node.resize(vertices,0);
    }
}

void MembraneMC::OutputTimes() {
    // Output time analysis
}
