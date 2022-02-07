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
#include "simulation.hpp"
using namespace std;


void Simulation::CheckerboardMCSweep(bool nl_move) {
// Implementation idea
// Working in two dimensions for decomposition
// Why two instead of three?
// Makes more sense as that's how the cell be 
// So for that, the space will be divided per
//  ||| 2 ||| ||| 3 |||
//  ||| 0 ||| ||| 1 |||
// Repeated for the whole system
// Constraints on checkboard x/y: must be divisble by 2
// Will have an ideal size >> 1, and then round to nearest even number from there
// For the use of the checkerboard itself, use the following
// ALGORITHM
// Shuffle order of checkerboard C that will be iterated through
// For cells c in C(i) do
//  shuffle particle ordering in c(j)
//  Select random particle in c(j), perform random move
//  Reject move if it goes out of cell
// end for
// Shift cell and rebuilt this list
    chrono::steady_clock::time_point t1_displace;
    chrono::steady_clock::time_point t2_displace;
    // Generate checkerboard
    t1_displace = chrono::steady_clock::now();
    generateCheckerboard(); 
    t2_displace = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_displace-t1_displace;
    time_storage_displace[0] += time_span.count();
    // Create order of C
    t1_displace = chrono::steady_clock::now();
    vector<int> set_order = {0, 1, 2, 3};
    shuffle_saru(generator, set_order);
    // Pick number of moves to do per sweep
    // Let's go with 3 times average number of particles per cell
    int move_count = (3*vertices)/(checkerboard_x*checkerboard_y);
    // Int array that has standard modifications depending on what set we are working on
    int cell_modify_x[4] = {0,1,0,1};
    int cell_modify_y[4] = {0,0,1,1};
    // Have diff arrays to soter results to avoid atomic operations
    // Set to 0 here
    #pragma omp parallel for
    for(int i=0; i<active_threads; i++) {
        Phi_diff_thread[i][0] = 0;
        Phi_bending_diff_thread[i][0] = 0;
        Phi_phi_diff_thread[i][0] = 0;
        Area_diff_thread[i][0] = 0;
        Mass_diff_thread[i][0] = 0;
        Magnet_diff_thread[i][0] = 0;
        steps_tested_displace_thread[i][0] = 0;
        steps_rejected_displace_thread[i][0] = 0;
        steps_tested_tether_thread[i][0] = 0;
        steps_rejected_tether_thread[i][0] = 0;
        steps_tested_mass_thread[i][0] = 0;
        steps_rejected_mass_thread[i][0] = 0;
        steps_tested_protein_thread[i][0] = 0;
        steps_rejected_protein_thread[i][0] = 0;
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    time_storage_displace[1] += time_span.count();
    // Now iterate through elements of checkerboard_set
    t1_displace = chrono::steady_clock::now();
    for(int i=0; i<4; i++) {
        // Loop through cells in set_order[i]
        // Note by construction checkerboard_x*checkerboard_y/4 elements
        #pragma omp parallel for
        for(int j=0; j<checkerboard_x*checkerboard_y/4; j++) { 
            Saru& local_generator = generators[omp_get_thread_num()];
            int thread_id = omp_get_thread_num();
            // Figure out which cell this one is working on
            // Have base of
            int cell_x = j%(checkerboard_x/2);
            int cell_y = j/(checkerboard_x/2);
            // Now use cell modify arrays to figure out what cell we on
            int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*checkerboard_x;
            // Now select random particles and perform random moves on them
            // Use new distribution functions to do the random selection
            // Determine nearest power to cell_current size
            unsigned int cell_size = checkerboard_list[cell_current].size();
            cell_size = cell_size - 1;
            cell_size = cell_size | (cell_size >> 1);
            cell_size = cell_size | (cell_size >> 2);
            cell_size = cell_size | (cell_size >> 4);
            cell_size = cell_size | (cell_size >> 8);
            cell_size = cell_size | (cell_size >> 16);
            for(int k=0; k<move_count; k++) {
                // Random particle
                int vertex_trial = local_generator.rand_select(checkerboard_list[cell_current].size()-1, cell_size);
                // cout << vertex_trial << endl;
                // cout << k << endl;
                // cout << cell_current << endl;
                // cout << checkerboard_list[cell_current].size() << endl;
                vertex_trial = checkerboard_list[cell_current][vertex_trial];
                // Random move
                double Chance = local_generator.d();
                if(Chance < 1.0/3.0) {
                    DisplaceStep(vertex_trial, thread_id);
                }
                else if ((Chance >= 1.0/3.0) && (Chance < 2.0/3.0)) {
                    TetherCut(vertex_trial, thread_id);
                }
                else {
                    ChangeMassNonCon(vertex_trial, thread_id);
                }
            }
            // dumpXYZCheckerboard("config_checker.xyz");
        }
    }
    // Now do nonlocal protein moves
    if (nl_move) {
        for(int i=0; i<4; i++) {
            // Construct pairs
            // Do so by shuffling a list of all cells
            vector<int> cells_possible;
            for(int j=0; j<checkerboard_x*checkerboard_y/4; j++) {
                // Figure out which cell this one is working on
                // Have base of
                int cell_x = j%(checkerboard_x/2);
                int cell_y = j/(checkerboard_x/2);
                // Now use cell modify arrays to figure out what cell we on
                int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*checkerboard_x;
                cells_possible.push_back(cell_current);
            }
            shuffle_saru(generator, cells_possible);
            #pragma omp parallel for
            for(int j=0; j<cells_possible.size()/2; j++) { 
                Saru& local_generator = generators[omp_get_thread_num()];
                int thread_id = omp_get_thread_num();
                int cells_0 = cells_possible[2*j];
                int cells_1 = cells_possible[2*j+1];
                // Determine nearest power to cell_current size
                unsigned int cell_size_0 = checkerboard_list[cells_0].size();
                cell_size_0 = cell_size_0 - 1;
                cell_size_0 = cell_size_0 | (cell_size_0 >> 1);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 2);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 4);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 8);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 16);
                unsigned int cell_size_1 = checkerboard_list[cells_1].size();
                cell_size_1 = cell_size_1 - 1;
                cell_size_1 = cell_size_1 | (cell_size_1 >> 1);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 2);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 4);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 8);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 16);
                for(int k=0; k<move_count; k++) {
                    // Random particle
                    int vertex_trial = local_generator.rand_select(checkerboard_list[cells_0].size()-1, cell_size_0);
                    int vertex_trial_2 = local_generator.rand_select(checkerboard_list[cells_1].size()-1, cell_size_1);
                    // cout << vertex_trial << endl;
                    // cout << k << endl;
                    // cout << cell_current << endl;
                    // cout << checkerboard_list[cell_current].size() << endl;
                    vertex_trial = checkerboard_list[cells_0][vertex_trial];
                    vertex_trial_2 = checkerboard_list[cells_1][vertex_trial_2];
                    moveProtein_nl(vertex_trial, vertex_trial_2, thread_id);
                }
                // dumpXYZCheckerboard("config_checker.xyz");
            }
        }
    }
    #pragma omp parallel for reduction(+:Phi,Phi_bending,Phi_phi,Area_total,Mass,Magnet,steps_rejected_displace,steps_tested_displace,steps_rejected_tether,steps_tested_tether,steps_rejected_mass,steps_tested_mass,steps_rejected_protein,steps_tested_protein)
    for(int i=0; i<active_threads; i++) {
        Phi += Phi_diff_thread[i][0];
        Phi_bending += Phi_bending_diff_thread[i][0];
        Phi_phi += Phi_phi_diff_thread[i][0];
        Area_total += Area_diff_thread[i][0];
        Mass += Mass_diff_thread[i][0];
        Magnet += Magnet_diff_thread[i][0];
        steps_rejected_displace += steps_rejected_displace_thread[i][0];
        steps_tested_displace += steps_tested_displace_thread[i][0];
        steps_rejected_tether += steps_rejected_tether_thread[i][0];
        steps_tested_tether += steps_tested_tether_thread[i][0];
        steps_rejected_mass += steps_rejected_mass_thread[i][0];
        steps_tested_mass += steps_tested_mass_thread[i][0];
        steps_rejected_protein += steps_rejected_protein_thread[i][0];
        steps_tested_protein += steps_tested_protein_thread[i][0];
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    time_storage_displace[2] += time_span.count();
}

void Simulation::NextStepSerial() {
// Pick step at random with given frequencies.
    // sigma_i_total = 0.0;
	/*
    double Chance = rand0to1(generator);
    if(Chance <= 0.5) {
        DisplaceStep();
    }
    else {
        TetherCut();        
    }
	*/
    double area_chance = 1.0/double(vertices);
    double chance_else = (1.0-area_chance)/3.0;
    for(int i=0; i<3; i++) {
        double Chance = generator.d();
        if(Chance < 1.0*chance_else) {
	        DisplaceStep();
        }
        else if ((Chance >= 1.0*chance_else) && (Chance < 2.0*chance_else)) {
            TetherCut();
        }
        else if ((Chance >= 2.0*chance_else) && (Chance < 1.0-area_chance)) {
            ChangeMassNonCon();
        }
        else {
            ChangeArea();
        }
    }
  // ChangeMassCon_nl();   
    // ChangeMassNonCon();
    // ChangeMassNonCon_gl();
    /*
	for(int i=0; i<10; i++) {
	    ChangeMassCon();
	}
    */
    // ChangeMassCon_nl();
    // cout << "Sigma_i_total is now " << sigma_i_total << endl;
}

void Simulation::NextStepParallel(bool nl_move) {
// Pick step at random with given frequencies.
// Only two options for now is Checkerboard Sweep or Area change
    double area_chance = 0.5;
    double Chance = generator.d();
    if(Chance < (1-area_chance)) {
        t1 = chrono::steady_clock::now();
        CheckerboardMCSweep(nl_move);
        t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        time_storage_cycle[0] += time_span.count();
    }
    else {
        t1 = chrono::steady_clock::now();
        ChangeArea();
        t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        time_storage_cycle[1] += time_span.count();
    }
}

void Simulation::Equilibriate(int cycles, chrono::steady_clock::time_point& begin) {
// Simulate for number of steps
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    double spon_curv_step = 4*spon_curv_end/cycles;
    int i=0;
    while(time_span_m.count() < final_warning) {
        SaruSeed(count_step);
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

void Simulation::Simulate(int cycles, chrono::steady_clock::time_point& begin) {
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
        SaruSeed(count_step);
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
