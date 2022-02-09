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

Simulation::Simulation(MembraneMC* sys_) {
    // Constructor
    // Assign system to current system
    sys = sys_;
}

Simulation::~Simulation() {
    // Destructor
    // Does nothing
}

void Simulation::CheckerboardMCSweep(bool nl_move) {
    // Implementation idea
    // Working in two dimensions for decomposition
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
    sys->nl->GenerateCheckerboard(); 
    t2_displace = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_displace-t1_displace;
    sys->time_storage_displace[0] += time_span.count();
    // Create order of C
    t1_displace = chrono::steady_clock::now();
    vector<int> set_order = {0, 1, 2, 3};
    sys->sim_util->ShuffleSaru(sys->generator, set_order);
    // Pick number of moves to do per sweep
    // Let's go with 3 times average number of particles per cell
    int move_count = (3*sys->vertices)/(sys->nl->checkerboard_x*sys->nl->checkerboard_y);
    // Int array that has standard modifications depending on what set we are working on
    int cell_modify_x[4] = {0,1,0,1};
    int cell_modify_y[4] = {0,0,1,1};
    // Have diff arrays to soter results to avoid atomic operations
    // Set to 0 here
    #pragma omp parallel for
    for(int i=0; i<sys->active_threads; i++) {
        phi_diff_thread[i][0] = 0;
        phi_bending_diff_thread[i][0] = 0;
        phi_phi_diff_thread[i][0] = 0;
        area_diff_thread[i][0] = 0;
        mass_diff_thread[i][0] = 0;
        magnet_diff_thread[i][0] = 0;
        sys->mc_mover->steps_tested_displace_thread[i][0] = 0;
        sys->mc_mover->steps_rejected_displace_thread[i][0] = 0;
        sys->mc_mover->steps_tested_tether_thread[i][0] = 0;
        sys->mc_mover->steps_rejected_tether_thread[i][0] = 0;
        sys->mc_mover->steps_tested_mass_thread[i][0] = 0;
        sys->mc_mover->steps_rejected_mass_thread[i][0] = 0;
        sys->mc_mover->steps_tested_protein_thread[i][0] = 0;
        sys->mc_mover->steps_rejected_protein_thread[i][0] = 0;
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    sys->time_storage_displace[1] += time_span.count();
    // Now iterate through elements of checkerboard_set
    t1_displace = chrono::steady_clock::now();
    for(int i=0; i<4; i++) {
        // Loop through cells in set_order[i]
        // Note by construction checkerboard_x*checkerboard_y/4 elements
        #pragma omp parallel for
        for(int j=0; j<sys->nl->checkerboard_x*sys->nl->checkerboard_y/4; j++) { 
            Saru& local_generator = sys->generators[omp_get_thread_num()];
            int thread_id = omp_get_thread_num();
            // Figure out which cell this one is working on
            // Have base of
            int cell_x = j%(sys->nl->checkerboard_x/2);
            int cell_y = j/(sys->nl->checkerboard_y/2);
            // Now use cell modify arrays to figure out what cell we on
            int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*checkerboard_x;
            // Now select random particles and perform random moves on them
            // Use new distribution functions to do the random selection
            // Determine nearest power to cell_current size
            unsigned int cell_size = sys->nl->checkerboard_list[cell_current].size();
            cell_size = cell_size - 1;
            cell_size = cell_size | (cell_size >> 1);
            cell_size = cell_size | (cell_size >> 2);
            cell_size = cell_size | (cell_size >> 4);
            cell_size = cell_size | (cell_size >> 8);
            cell_size = cell_size | (cell_size >> 16);
            for(int k=0; k<move_count; k++) {
                // Random particle
                int vertex_trial = local_generator.rand_select(sys->nl->checkerboard_list[cell_current].size()-1, cell_size);
                vertex_trial = sys->nl->checkerboard_list[cell_current][vertex_trial];
                // Random move
                double chance = local_generator.d();
                if(chance < 1.0/3.0) {
                    sys->mc_mover->DisplaceStep(vertex_trial, thread_id);
                }
                else if ((chance >= 1.0/3.0) && (chance < 2.0/3.0)) {
                    sys->mc_mover->TetherCut(vertex_trial, thread_id);
                }
                else {
                    sys->mc_mover->ChangeMassNonCon(vertex_trial, thread_id);
                }
            }
        }
    }
    // Now do nonlocal protein moves
    if (nl_move) {
        for(int i=0; i<4; i++) {
            // Construct pairs
            // Do so by shuffling a list of all cells
            vector<int> cells_possible;
            for(int j=0; j<sys->nl->checkerboard_x*sys->nl->checkerboard_y/4; j++) {
                // Figure out which cell this one is working on
                // Have base of
                int cell_x = j%(sys->nl->checkerboard_x/2);
                int cell_y = j/(sys->nl->checkerboard_y/2);
                // Now use cell modify arrays to figure out what cell we on
                int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*checkerboard_x;
                cells_possible.push_back(cell_current);
            }
            shuffle_saru(sys->generator, cells_possible);
            #pragma omp parallel for
            for(int j=0; j<cells_possible.size()/2; j++) { 
                Saru& local_generator = sys->generators[omp_get_thread_num()];
                int thread_id = omp_get_thread_num();
                int cells_0 = cells_possible[2*j];
                int cells_1 = cells_possible[2*j+1];
                // Determine nearest power to cell_current size
                unsigned int cell_size_0 = sys->nl->checkerboard_list[cells_0].size();
                cell_size_0 = cell_size_0 - 1;
                cell_size_0 = cell_size_0 | (cell_size_0 >> 1);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 2);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 4);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 8);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 16);
                unsigned int cell_size_1 = sys->nl->checkerboard_list[cells_1].size();
                cell_size_1 = cell_size_1 - 1;
                cell_size_1 = cell_size_1 | (cell_size_1 >> 1);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 2);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 4);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 8);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 16);
                for(int k=0; k<move_count; k++) {
                    // Random particle
                    int vertex_trial = local_generator.rand_select(sys->nl->checkerboard_list[cells_0].size()-1, cell_size_0);
                    int vertex_trial_2 = local_generator.rand_select(sys->nl->checkerboard_list[cells_1].size()-1, cell_size_1);
                    vertex_trial = sys->nl->checkerboard_list[cells_0][vertex_trial];
                    vertex_trial_2 = sys->nl->checkerboard_list[cells_1][vertex_trial_2];
                    sys->mc_mover->MoveProteinNL(vertex_trial, vertex_trial_2, thread_id);
                }
            }
        }
    }
    #pragma omp parallel for reduction(+:sys->phi,sys->phi_bending,sys->phi_phi,area->area_total,sys->mass,sys->magnet,sys->mc_mover->steps_rejected_displace,sys->mc_mover->steps_tested_displace,sys->mc_mover->steps_rejected_tether,sys->mc_mover->steps_tested_tether,sys->mc_mover->steps_rejected_mass,sys->mc_mover->steps_tested_mass,sys->mc_mover->steps_rejected_protein,sys->mc_mover->steps_tested_protein)
    for(int i=0; i<sys->active_threads; i++) {
        sys->phi += phi_diff_thread[i][0];
        sys->phi_bending += phi_bending_diff_thread[i][0];
        sys->phi_phi += phi_phi_diff_thread[i][0];
        area->area_total += Area_diff_thread[i][0];
        sys->mass += mass_diff_thread[i][0];
        sys->magnet += magnet_diff_thread[i][0];
        sys->mc_mover->steps_rejected_displace += steps_rejected_displace_thread[i][0];
        sys->mc_mover->steps_tested_displace += steps_tested_displace_thread[i][0];
        sys->mc_mover->steps_rejected_tether += steps_rejected_tether_thread[i][0];
        sys->mc_mover->steps_tested_tether += steps_tested_tether_thread[i][0];
        sys->mc_mover->steps_rejected_mass += steps_rejected_mass_thread[i][0];
        sys->mc_mover->steps_tested_mass += steps_tested_mass_thread[i][0];
        sys->mc_mover->steps_rejected_protein += steps_rejected_protein_thread[i][0];
        sys->mc_mover->steps_tested_protein += steps_tested_protein_thread[i][0];
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    sys->time_storage_displace[2] += time_span.count();
}

void Simulation::NextStepSerial() {
    // Pick step at random with given frequencies.
    double area_chance = 1.0/double(sys->vertices);
    double chance_else = (1.0-area_chance)/3.0;
    for(int i=0; i<3; i++) {
        double Chance = sys->generator.d();
        if(Chance < 1.0*chance_else) {
	        sys->mc_mover->DisplaceStep();
        }
        else if ((Chance >= 1.0*chance_else) && (Chance < 2.0*chance_else)) {
            sys->mc_mover->TetherCut();
        }
        else if ((Chance >= 2.0*chance_else) && (Chance < 1.0-area_chance)) {
            sys->mc_mover->ChangeMassNonCon();
        }
        else {
            sys->mc_mover->ChangeArea();
        }
    }
}

void Simulation::NextStepParallel(bool nl_move) {
    // Pick step at random with given frequencies.
    // Only two options for now area checkerboard sweep or area change
    double area_chance = 0.5;
    double Chance = sys->generator.d();
    if(Chance < (1-area_chance)) {
        t1 = chrono::steady_clock::now();
        CheckerboardMCSweep(nl_move);
        t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        sys->time_storage_cycle[0] += time_span.count();
    }
    else {
        t1 = chrono::steady_clock::now();
        sys->mc_mover->ChangeArea();
        t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        sys->time_storage_cycle[1] += time_span.count();
    }
}
