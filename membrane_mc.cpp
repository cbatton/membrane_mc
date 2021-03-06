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

    // Start clock
    t1 = chrono::steady_clock::now();
    // Check to see if we can restart
    ifstream input_2;
    input_2.open(output_path+"/int.off");
    // Check to see if int.off present. If not, do nothing
    if (input_2.fail()) {
        if(world_rank == 0) {
            my_cout << "No restart file." << endl;
        }
    }
    else{
        if(world_rank == 0) {
            my_cout << "Restart file found." << endl;
        }
        restart = 1;
    }
    // Now read in param file
    ifstream input;
    input.open(output_path+"/param");
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        if(world_rank == 0) {
            cout << "No input file." << endl;
        }
    }
    else{
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
        getline(input, line);
        input >> line >> dump_cycle >> dump_int >> dump_int_2 >> dump_config;
        input.close();
        // Output variables for catching things
        if((world_rank == 0) && (restart == 0)) {
            my_cout << "Param file detected. Changing values." << endl;
            my_cout << "Dimensions is now " << dim_x << " " << dim_y << endl;
            my_cout << "Cycles is now " << cycles_eq << " " << cycles_prod << endl;
            my_cout << "Lengths is now " << lengths[0] << " " << lengths[1] << " " << lengths[2] << endl;
            my_cout << "Temperature is now " << temp_list[0] << " " << temp_list[1] << endl;
            my_cout << "k_b is now " << k_b[0] << " " << k_b[1] << " " << k_b[2] << endl;
            my_cout << "gamma_surf is now " << gamma_surf[0] << " " << gamma_surf[1] << " " << gamma_surf[2] << endl;
            my_cout << "tau_frame is now " << tau_frame << endl;
            my_cout << "Spontaneous curvature is now " << spon_curv[0] << " " << spon_curv[1] << " " << spon_curv[2] << endl;
            my_cout << "Lengths is now " << lengths[0] << " " << lengths[1] << " " << lengths[2] << endl;
            my_cout << "lambda is now " << lambda << endl;
            my_cout << "lambda_scale is now " << lambda_scale << endl;
            my_cout << "ising_values is now " << ising_values[0] << " " << ising_values[1] << " " << ising_values[2] << endl;
            my_cout << "j_coupling is now " << j_coupling[0][0] << " " << j_coupling[0][1] << " " << j_coupling[0][2] << endl;
            my_cout << "\t" << j_coupling[1][0] << " " << j_coupling[1][1] << " " << j_coupling[1][2] << endl;
            my_cout << "\t" << j_coupling[2][0] << " " << j_coupling[2][1] << " " << j_coupling[2][2] << endl;
            my_cout << "h is now " << h_external << endl;
            my_cout << "num_frac is now " << num_frac << endl;
            my_cout << "num_proteins is now " << num_proteins << endl;
            my_cout << "seed_base is now " << seed_base << endl;
            my_cout << "count_step is now " << count_step << endl;
            my_cout << "final_time is now " << final_time << endl;
            my_cout << "nl_move_start is now " << nl_move_start << endl;
            my_cout << "bins is now " << bins << endl;
            my_cout << "dump frequency is now " << dump_cycle << " " << dump_int << " " << dump_int_2 << " " << dump_config << endl;
        }
        // Set lengths and seed_base
        lengths_old = lengths;
        lengths_base = lengths;
        // Hash seed_base
        seed_base = seed_base*0x12345677 + 0x12345;
        seed_base = seed_base^(seed_base>>16);
        seed_base = seed_base*0x45679;
        if(world_rank == 0) {
            my_cout << "seed_base is now " << seed_base << endl;
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

void MembraneMC::OutputTimes(chrono::steady_clock::time_point& begin) {
    // Begin by figuring out the total time
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> time_span = end-begin;
    // Output time analysis
    my_cout << "Total time: " << time_span.count() << " s" << endl;
    my_cout << "Checkerboard MC time: " << time_storage_cycle[0] << " s" << endl;
    my_cout << "Area MC time: " << time_storage_cycle[1] << " s" << endl;
    my_cout << "Other time: " << time_span.count()-time_storage_cycle[0]-time_storage_cycle[1] << " s" << endl;

    my_cout << "\nDisplace MC breakdown" << endl;
    double displace_storage_total = 0;
    for(int i=0; i<3; i++) {
        displace_storage_total += time_storage_displace[i];
    }
    my_cout << "Checkerboard: " << time_storage_displace[0] << " s, " << time_storage_displace[0]/time_storage_cycle[1] << " per" << endl;
    my_cout << "Shuffling: " << time_storage_displace[1] << " s, " << time_storage_displace[1]/time_storage_cycle[1] << " per" << endl;
    my_cout << "Steps: " << time_storage_displace[2] << " s, " << time_storage_displace[2]/time_storage_cycle[1] << " per" << endl;
    my_cout << "Misc: " << time_storage_cycle[0]-displace_storage_total << " s, " << (time_storage_cycle[0]-displace_storage_total)/time_storage_cycle[0] << " per" << endl;

    my_cout << "\nArea MC breakdown" << endl;
    double area_storage_total = 0;
    for(int i=0; i<4; i++) {
        area_storage_total += time_storage_area[i];
    }
    my_cout << "Initial: " << time_storage_area[0] << " s, " << time_storage_area[0]/time_storage_cycle[1] << " per" << endl;
    my_cout << "Neighbor list: " << time_storage_area[1] << " s, " << time_storage_area[1]/time_storage_cycle[1] << " per" << endl;
    my_cout << "Energy calculation: " << time_storage_area[2] << " s, " << time_storage_area[2]/time_storage_cycle[1] << " per" << endl;
    my_cout << "A/R: " << time_storage_area[3] << " s, " << time_storage_area[3]/time_storage_cycle[1] << " per" << endl;
    my_cout << "Misc: " << time_storage_cycle[1]-area_storage_total << " s, " << (time_storage_cycle[1]-area_storage_total)/time_storage_cycle[1] << " per" << endl;

    double other_storage_total = 0;
    for(int i=0; i<6; i++) {
        other_storage_total += time_storage_other[i];
    }
    my_cout << "\nOther breakdown" << endl;
    my_cout << "Generating system: " << time_storage_other[0] <<" s, " << time_storage_other[0]/other_storage_total << " per" << endl;
    my_cout << "Initial cycle output: " << time_storage_other[1] <<" s, " << time_storage_other[1]/other_storage_total << " per" << endl;
    my_cout << "Distance check: " << time_storage_other[2] <<" s, " << time_storage_other[2]/other_storage_total << " per" << endl;
    my_cout << "Output reject per: " << time_storage_other[3] <<" s, " << time_storage_other[3]/other_storage_total << " per" << endl;
    my_cout << "Output configurations: " << time_storage_other[4] <<" s, " << time_storage_other[4]/other_storage_total << " per" << endl;
    my_cout << "Analyzers: " << time_storage_other[5] <<" s, " << time_storage_other[5]/other_storage_total << " per" << endl;
    my_cout << "Misc: " << time_span.count()-time_storage_cycle[0]-time_storage_cycle[1]-other_storage_total <<" s";
}
