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
#include "analyzers.hpp"
#include "init_system.hpp"
#include "mc_moves.hpp"
#include "neighborlist.hpp"
#include "output_system.hpp"
#include "simulation.hpp"
#include "sim_utilities.hpp"
#include "utilities.hpp"
using namespace std;

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

    // Initialize OpenMP
    active_threads = omp_get_max_threads();

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
        int storage_neighor = 10;
        int storage_umb_time = 100;
        // Read in everything
        input >> line >> dim_x >> dim_y >> endl;
        getline(input,line);
        input >> line >> cycles_eq >> cycles_prod;
        getline(input, line);
        input >> line >> storage_time >> storage_neighbor >> storage_umb_time;
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
        input >> line >> Length_x >> Length_y >> Length_z;
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
            cout << "scale is now " << scale << endl;
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
        initializer->SaruSeed(count_step);
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
        area_vertex.resize(vertices,0.0);
        area_vertex_original.resize(vertices,0.0);
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
        // Now Initialize utility classes
        analysis = make_shared<analyzers>(this, bins, storage_time, storage_neighor, storage_umb_time);
        mc_mover = make_shared<mc_moves>;
        nl = make_shared<neighborlist>;
        output = make_shared<output_system>;
        sim = make_shared<simulation>;
        sim_util = make_shared<sim_utilities>;
        util = make_shared<utilities>;
        // Storage variables
        storage = Cycles_prod/storage_time;
        energy_storage.resize(storage,0.0);
        area_storage.resize(storage,0.0);
        area_proj_storage.resize(storage,0.0);
        mass_storage.resize(storage,0.0);
        vector<long long int> list_long;
        numbers_neighbor.resize(neighbor_max,list_long);
        for(int i=0; i<neighbor_max; i++) {
            numbers_neighbor[i].resize(Cycles_prod/storage_neighbor);
        }
    }
}

void MembraneMC::Equilibriate(int cycles, chrono::steady_clock::time_point& begin) {
// Simulate for number of steps
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    double spon_curv_step = 4*spon_curv_end/cycles;
    int i=0;
    while(time_span_m.count() < final_warning) {
        initializer->SaruSeed(count_step);
        count_step++;
        if(nl_move_start == 0) {
            nextStepParallel(false);
        }
        else {
            nextStepParallel(true);
        }
        if(i < cycles/4) {
            spon_curv[2] += spon_curv_step;
            initializeEnergy();
        }
        else if(i == cycles/4){
            spon_curv[2] = spon_curv_end;
        }
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
            double Phi_ = Phi;
            double Phi_bending_ = Phi_bending;
            double Phi_phi_ = Phi_phi;
            initializeEnergy();
            cout.rdbuf(myfilebuf);
			cout << "Cycle " << i << endl;
			cout << "Energy " << std::scientific << Phi << " " << std::scientific << Phi-Phi_ << endl;
            cout << "Phi_bending " << std::scientific << Phi_bending << " " << std::scientific << Phi_bending-Phi_bending_ << " Phi_phi " << std::scientific << Phi_phi << " " << std::scientific << Phi_phi-Phi_phi_ << endl;
            cout << "Mass " << Mass << endl;
            cout << "Area " << Area_total << " and " << Length_x*Length_y << endl;
            cout << "spon_curv " << spon_curv[2] << endl;
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[0] += time_span.count();

            t1_other = chrono::steady_clock::now();
			linkMaxMin();
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[1] += time_span.count();

            t1_other = chrono::steady_clock::now();
            cout << "Displace " << steps_rejected_displace << "/" << steps_tested_displace << endl;
            cout << "Tether " << steps_rejected_tether << "/" << steps_tested_tether << endl;
            cout << "Mass " << steps_rejected_mass << "/" << steps_tested_mass << endl;
            cout << "Protein " << steps_rejected_protein << "/" << steps_tested_protein << endl;
            cout << "Area " << steps_rejected_area << "/" << steps_tested_area << endl;
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[2] += time_span.count();
            // cout << "Percentage of rejected steps: " << steps_rejected_displace+steps_rejected_tether << "/" << steps_tested_displace+steps_tested_tether << endl;
			/*
			cout << "Max diff is " << max_diff << endl;
			cout << "Relative difference is " << relative_diff << endl;
			max_diff = -1;
			relative_diff = 0;
			*/
		}
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
			outputTriangulation("int.off");	
            if(i%40000==0) {
                outputTriangulation("int_2.off");	
            }
            if(i%4000==0) {
			    dumpXYZConfig("config_equil.xyz");
			    // dumpXYZConfigNormal("config_equil_normal.xyz");
			    // dumpXYZCheckerboard("config_equil_test.xyz");
			    outputTriangulationAppend("equil.off");	
            }
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[3] += time_span.count();
		}
        /*
        if(i%100000==0) {
            outputTriangulationStorage();
        }
		if(i%40000==0) {
			dumpPhiNode("phinode_equil.txt");
            dumpAreaNode("areanode_equil.txt");
		}
        */
        middle = chrono::steady_clock::now();
        time_span_m = middle-begin;
        i++;
        if(i >= cycles) {
            break;
        }
    }
    outputTriangulation("int.off");	
    steps_tested_eq = steps_tested_displace + steps_tested_tether + steps_tested_mass + steps_tested_protein + steps_tested_area;
    steps_rejected_eq = steps_rejected_displace + steps_tested_tether + steps_rejected_mass + steps_rejected_protein + steps_rejected_area;
}

void MembraneMC::Simulate(int cycles, chrono::steady_clock::time_point& begin) {
// Simulate for number of cycles
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    steps_tested_displace = 0;
    steps_rejected_displace = 0;
    steps_tested_tether = 0;
    steps_rejected_tether = 0;
    steps_tested_mass = 0;
    steps_rejected_mass = 0;
    steps_tested_protein = 0;
    steps_rejected_protein = 0;
    steps_tested_area = 0;
    steps_rejected_area = 0;
    ofstream myfile_umb;
    myfile_umb.precision(17);
    myfile_umb.open(output_path+"/mbar_data.txt", std::ios_base::app);
    cout.precision(8);
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    // cout << "Total time: " << time_span.count() << " s" << endl;
    int i = 0;
    // for(int i=0; i<cycles; i++) {
    while(time_span_m.count() < final_warning) {
        initializer->SaruSeed(count_step);
        count_step++;
	    nextStepParallel(true);
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
            double Phi_ = Phi;
            double Phi_bending_ = Phi_bending;
            double Phi_phi_ = Phi_phi;
            initializeEnergy();
            cout.rdbuf(myfilebuf);
			cout << "Cycle " << i << endl;
			cout << "Energy " << std::scientific << Phi << " " << std::scientific << Phi-Phi_ << endl;
            cout << "Phi_bending " << std::scientific << Phi_bending << " " << std::scientific << Phi_bending-Phi_bending_ << " Phi_phi " << std::scientific << Phi_phi << " " << std::scientific << Phi_phi-Phi_phi_ << endl;
            cout << "Mass " << Mass << endl;
            cout << "Area " << Area_total << " and " << Length_x*Length_y << endl;
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[0] += time_span.count();

            t1_other = chrono::steady_clock::now();
			linkMaxMin();
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[1] += time_span.count();

            t1_other = chrono::steady_clock::now();
            cout << "Displace " << steps_rejected_displace << "/" << steps_tested_displace << endl;
            cout << "Tether " << steps_rejected_tether << "/" << steps_tested_tether << endl;
            cout << "Mass " << steps_rejected_mass << "/" << steps_tested_mass << endl;
            cout << "Protein " << steps_rejected_protein << "/" << steps_tested_protein << endl;
            cout << "Area " << steps_rejected_area << "/" << steps_tested_area << endl;
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[2] += time_span.count();
            // cout << "Percentage of rejected steps: " << steps_rejected_displace+steps_rejected_tether << "/" << steps_tested_displace+steps_tested_tether << endl;
			/*
			cout << "Max diff is " << max_diff << endl;
			cout << "Relative difference is " << relative_diff << endl;
			max_diff = -1;
			relative_diff = 0;
			*/
		}
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
			outputTriangulation("int.off");	
            if(count_step%20000==0) {
                outputTriangulation("int_2.off");	
            }
            if(i%4000==0) {
			    dumpXYZConfig("config.xyz");
			    // dumpXYZConfigNormal("config_normal.xyz");
			    outputTriangulationAppend("prod.off");	
			    // dumpXYZCheckerboard("config_test.xyz");
            }
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[3] += time_span.count();
		}
        /*
        if(i%100000==0) {
            outputTriangulationStorage();
        }
		if(i%40000==0) {
			dumpPhiNode("phinode.txt");
            dumpAreaNode("areanode.txt");
		}
        */
        if(i%storage_neighbor==0) {
            // sampleNumberNeighbors(i/storage_neighbor);  
            // dumpNumberNeighbors("numbers_dump.txt", i/storage_neighbor);
            // neighbors_counts++;
        }
        if(i%storage_umb_time==0) {
            energy_storage_umb[i/storage_umb_time] = Phi;
            umbOutput(i/storage_umb_time, myfile_umb);
            umb_counts++;
        }
        if(i%storage_time==0) {
		    energy_storage[i/storage_time] = Phi;
            area_storage[i/storage_time] = Area_total;
            area_proj_storage[i/storage_time] = Length_x*Length_y;
            mass_storage[i/storage_time] = Mass;
            storage_counts++;
        }
        middle = chrono::steady_clock::now();
        time_span_m = middle-begin;
        i++;
        if(i >= cycles) {
            break;
        }
        // dumpXYZConfig("config.xyz");
    }
    outputTriangulation("int.off");	
    steps_tested_prod = steps_tested_displace + steps_tested_tether + steps_tested_mass + steps_tested_protein + steps_tested_area;
    steps_rejected_prod = steps_rejected_displace + steps_tested_tether + steps_rejected_mass + steps_rejected_protein + steps_rejected_area;
    myfile_umb.close();
}

void MembraneMC::OutputTimes() {
    // Output time analysis
}

void MembraneMC::OutputAnalyzers() {
    // Output analyzers
}
