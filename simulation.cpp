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
#include "analyzers.hpp"
#include "output_system.hpp"
#include "mc_moves.hpp"
#include "utilities.hpp"
using namespace std;

Simulation::Simulation(double lambda, double lambda_scale, int nl_move_start, int max_threads) : mc_mover(lambda, lambda_scale, nl_move_start, max_threads) {
    // Constructor
    // Does nothing
}

Simulation::~Simulation() {
    // Destructor
    // Does nothing
}

void Simulation::CheckerboardMCSweep(bool nl_move, MembraneMC& sys, NeighborList& nl) {
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
    nl.GenerateCheckerboard(sys); 
    t2_displace = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_displace-t1_displace;
    sys.time_storage_displace[0] += time_span.count();
    // Create order of C
    t1_displace = chrono::steady_clock::now();
    vector<int> set_order = {0, 1, 2, 3};
    util.ShuffleSaru(sys.generator, set_order);
    // Pick number of moves to do per sweep
    // Let's go with 3 times average number of particles per cell
    int move_count = (3*sys.vertices)/(nl.checkerboard_x*nl.checkerboard_y);
    // Int array that has standard modifications depending on what set we are working on
    int cell_modify_x[4] = {0,1,0,1};
    int cell_modify_y[4] = {0,0,1,1};
    // Have diff arrays to soter results to avoid atomic operations
    // Set to 0 here
    #pragma omp parallel for
    for(int i=0; i<sys.active_threads; i++) {
        sys.phi_diff_thread[i][0] = 0;
        sys.phi_bending_diff_thread[i][0] = 0;
        sys.phi_phi_diff_thread[i][0] = 0;
        sys.area_diff_thread[i][0] = 0;
        sys.mass_diff_thread[i][0] = 0;
        sys.magnet_diff_thread[i][0] = 0;
        mc_mover.steps_tested_displace_thread[i][0] = 0;
        mc_mover.steps_rejected_displace_thread[i][0] = 0;
        mc_mover.steps_tested_tether_thread[i][0] = 0;
        mc_mover.steps_rejected_tether_thread[i][0] = 0;
        mc_mover.steps_tested_mass_thread[i][0] = 0;
        mc_mover.steps_rejected_mass_thread[i][0] = 0;
        mc_mover.steps_tested_protein_thread[i][0] = 0;
        mc_mover.steps_rejected_protein_thread[i][0] = 0;
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    sys.time_storage_displace[1] += time_span.count();
    // Now iterate through elements of checkerboard_set
    t1_displace = chrono::steady_clock::now();
    for(int i=0; i<4; i++) {
        // Loop through cells in set_order[i]
        // Note by construction checkerboard_x*checkerboard_y/4 elements
        #pragma omp parallel for
        for(int j=0; j<nl.checkerboard_x*nl.checkerboard_y/4; j++) { 
            Saru& local_generator = sys.generators[omp_get_thread_num()];
            int thread_id = omp_get_thread_num();
            // Figure out which cell this one is working on
            // Have base of
            int cell_x = j%(nl.checkerboard_x/2);
            int cell_y = j/(nl.checkerboard_y/2);
            // Now use cell modify arrays to figure out what cell we on
            int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*nl.checkerboard_x;
            // Now select random particles and perform random moves on them
            // Use new distribution functions to do the random selection
            // Determine nearest power to cell_current size
            unsigned int cell_size = nl.checkerboard_list[cell_current].size();
            cell_size = cell_size - 1;
            cell_size = cell_size | (cell_size >> 1);
            cell_size = cell_size | (cell_size >> 2);
            cell_size = cell_size | (cell_size >> 4);
            cell_size = cell_size | (cell_size >> 8);
            cell_size = cell_size | (cell_size >> 16);
            for(int k=0; k<move_count; k++) {
                // Random particle
                int vertex_trial = local_generator.rand_select(nl.checkerboard_list[cell_current].size()-1, cell_size);
                vertex_trial = nl.checkerboard_list[cell_current][vertex_trial];
                // Random move
                double chance = local_generator.d();
                if(chance < 1.0/3.0) {
                    mc_mover.DisplaceStep(sys, nl, vertex_trial, thread_id);
                }
                else if ((chance >= 1.0/3.0) && (chance < 2.0/3.0)) {
                    mc_mover.TetherCut(sys, nl, vertex_trial, thread_id);
                }
                else {
                    mc_mover.ChangeMassNonCon(sys, nl, vertex_trial, thread_id);
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
            for(int j=0; j<nl.checkerboard_x*nl.checkerboard_y/4; j++) {
                // Figure out which cell this one is working on
                // Have base of
                int cell_x = j%(nl.checkerboard_x/2);
                int cell_y = j/(nl.checkerboard_y/2);
                // Now use cell modify arrays to figure out what cell we on
                int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*nl.checkerboard_x;
                cells_possible.push_back(cell_current);
            }
            util.ShuffleSaru(sys.generator, cells_possible);
            #pragma omp parallel for
            for(int j=0; j<cells_possible.size()/2; j++) { 
                Saru& local_generator = sys.generators[omp_get_thread_num()];
                int thread_id = omp_get_thread_num();
                int cells_0 = cells_possible[2*j];
                int cells_1 = cells_possible[2*j+1];
                // Determine nearest power to cell_current size
                unsigned int cell_size_0 = nl.checkerboard_list[cells_0].size();
                cell_size_0 = cell_size_0 - 1;
                cell_size_0 = cell_size_0 | (cell_size_0 >> 1);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 2);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 4);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 8);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 16);
                unsigned int cell_size_1 = nl.checkerboard_list[cells_1].size();
                cell_size_1 = cell_size_1 - 1;
                cell_size_1 = cell_size_1 | (cell_size_1 >> 1);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 2);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 4);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 8);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 16);
                for(int k=0; k<move_count; k++) {
                    // Random particle
                    int vertex_trial = local_generator.rand_select(nl.checkerboard_list[cells_0].size()-1, cell_size_0);
                    int vertex_trial_2 = local_generator.rand_select(nl.checkerboard_list[cells_1].size()-1, cell_size_1);
                    vertex_trial = nl.checkerboard_list[cells_0][vertex_trial];
                    vertex_trial_2 = nl.checkerboard_list[cells_1][vertex_trial_2];
                    mc_mover.MoveProteinNL(sys, vertex_trial, vertex_trial_2, thread_id);
                }
            }
        }
    }
    double phi = 0.0;
    double phi_bending = 0.0;
    double phi_phi = 0.0;
    double area_total = 0.0;
    int mass = 0;
    double magnet = 0;
    int steps_rejected_displace = 0;
    int steps_tested_displace = 0;
    int steps_rejected_tether = 0;
    int steps_tested_tether = 0;
    int steps_rejected_mass = 0;
    int steps_tested_mass = 0;
    int steps_rejected_protein = 0;
    int steps_tested_protein = 0;
    #pragma omp parallel for reduction(+:phi,phi_bending,phi_phi,area_total,mass,magnet,steps_rejected_displace,steps_tested_displace,steps_rejected_tether,steps_tested_tether,steps_rejected_mass,steps_tested_mass,steps_rejected_protein,steps_tested_protein)
    for(int i=0; i<sys.active_threads; i++) {
        phi += sys.phi_diff_thread[i][0];
        phi_bending += sys.phi_bending_diff_thread[i][0];
        phi_phi += sys.phi_phi_diff_thread[i][0];
        area_total += sys.area_diff_thread[i][0];
        mass += sys.mass_diff_thread[i][0];
        magnet += sys.magnet_diff_thread[i][0];
        steps_rejected_displace += mc_mover.steps_rejected_displace_thread[i][0];
        steps_tested_displace += mc_mover.steps_tested_displace_thread[i][0];
        steps_rejected_tether += mc_mover.steps_rejected_tether_thread[i][0];
        steps_tested_tether += mc_mover.steps_tested_tether_thread[i][0];
        steps_rejected_mass += mc_mover.steps_rejected_mass_thread[i][0];
        steps_tested_mass += mc_mover.steps_tested_mass_thread[i][0];
        steps_rejected_protein += mc_mover.steps_rejected_protein_thread[i][0];
        steps_tested_protein += mc_mover.steps_tested_protein_thread[i][0];
    }
    sys.phi += phi;
    sys.phi_bending += phi_bending;
    sys.phi_phi += phi_phi;
    sys.area_total += area_total;
    sys.mass += mass;
    sys.magnet += magnet;
    mc_mover.steps_rejected_displace += steps_rejected_displace;
    mc_mover.steps_tested_displace += steps_tested_displace;
    mc_mover.steps_rejected_tether += steps_rejected_tether;
    mc_mover.steps_tested_tether += steps_tested_tether;
    mc_mover.steps_rejected_mass += steps_rejected_mass;
    mc_mover.steps_tested_mass += steps_tested_mass;
    mc_mover.steps_rejected_protein += steps_rejected_protein;
    mc_mover.steps_tested_protein += steps_tested_protein;
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    sys.time_storage_displace[2] += time_span.count();
}

void Simulation::NextStepSerial(MembraneMC& sys, NeighborList& nl) {
    // Pick step at random with given frequencies.
    double area_chance = 1.0/double(sys.vertices);
    double chance_else = (1.0-area_chance)/3.0;
    for(int i=0; i<3; i++) {
        double Chance = sys.generator.d();
        if(Chance < 1.0*chance_else) {
            int vertex_trial = sys.generator.rand_select(sys.vertices-1);
	        mc_mover.DisplaceStep(sys,nl,vertex_trial,0);
        }
        else if ((Chance >= 1.0*chance_else) && (Chance < 2.0*chance_else)) {
            int vertex_trial = sys.generator.rand_select(sys.vertices-1);
            mc_mover.TetherCut(sys,nl,vertex_trial,0);
        }
        else if ((Chance >= 2.0*chance_else) && (Chance < 1.0-area_chance)) {
            int vertex_trial = sys.generator.rand_select(sys.vertices-1);
            mc_mover.ChangeMassNonCon(sys,nl,vertex_trial,0);
        }
        else {
            mc_mover.ChangeArea(sys,nl);
        }
    }
}

void Simulation::NextStepParallel(bool nl_move, MembraneMC& sys, NeighborList& nl) {
    // Pick step at random with given frequencies.
    // Only two options for now area checkerboard sweep or area change
    double area_chance = 0.5;
    double Chance = sys.generator.d();
    if(Chance < (1-area_chance)) {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        CheckerboardMCSweep(nl_move, sys, nl);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        sys.time_storage_cycle[0] += time_span.count();
    }
    else {
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        mc_mover.ChangeArea(sys, nl);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        sys.time_storage_cycle[1] += time_span.count();
    }
}

void Simulation::Equilibriate(int cycles, MembraneMC& sys, NeighborList& nl, chrono::steady_clock::time_point& begin) {
    // Simulate for number of steps, or time limit
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    double spon_curv_step = 4*sys.spon_curv_end/cycles;
    int i=0;
    while(time_span_m.count() < sys.final_warning) {
        util.SaruSeed(sys,sys.count_step);
        sys.count_step++;
        if(mc_mover.nl_move_start == 0) {
            NextStepParallel(false, sys, nl);
        }
        else {
            NextStepParallel(true, sys, nl);
        }
        if(i < cycles/4) {
            sys.spon_curv[2] += spon_curv_step;
            util.InitializeEnergy(sys, nl);
        }
        else if(i == cycles/4){
            sys.spon_curv[2] = sys.spon_curv_end;
        }
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
            double phi_ = sys.phi;
            double phi_bending_ = sys.phi_bending;
            double phi_phi_ = sys.phi_phi;
            util.InitializeEnergy(sys, nl);
			sys.my_cout << "cycle " << i << endl;
			sys.my_cout << "energy " << std::scientific << sys.phi << " " << std::scientific << sys.phi-phi_ << endl;
            sys.my_cout << "phi_bending " << std::scientific << sys.phi_bending << " " << std::scientific << sys.phi_bending-phi_bending_ << " phi_phi " << std::scientific << sys.phi_phi << " " << std::scientific << sys.phi_phi-phi_phi_ << endl;
            sys.my_cout << "mass " << sys.mass << endl;
            sys.my_cout << "area " << sys.area_total << " and " << sys.lengths[0]*sys.lengths[1] << endl;
            sys.my_cout << "spon_curv " << sys.spon_curv[2] << endl;
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            sys.time_storage_other[0] += time_span.count();

            t1_other = chrono::steady_clock::now();
			util.LinkMaxMin(sys, nl);
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            sys.time_storage_other[1] += time_span.count();

            t1_other = chrono::steady_clock::now();
            sys.my_cout << "displace " << mc_mover.steps_rejected_displace << "/" << mc_mover.steps_tested_displace << endl;
            sys.my_cout << "tether " << mc_mover.steps_rejected_tether << "/" << mc_mover.steps_tested_tether << endl;
            sys.my_cout << "mass " << mc_mover.steps_rejected_mass << "/" << mc_mover.steps_tested_mass << endl;
            sys.my_cout << "protein " << mc_mover.steps_rejected_protein << "/" << mc_mover.steps_tested_protein << endl;
            sys.my_cout << "area " << mc_mover.steps_rejected_area << "/" << mc_mover.steps_tested_area << endl;
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            sys.time_storage_other[2] += time_span.count();
		}
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
			output.OutputTriangulation(sys, "int.off");	
            if(i%40000==0) {
                output.OutputTriangulation(sys, "int_2.off");	
            }
            if(i%4000==0) {
			    output.DumpXYZConfig(sys, "config_equil.xyz");
			    output.OutputTriangulationAppend(sys, "equil.off");	
            }
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            sys.time_storage_other[3] += time_span.count();
		}
        middle = chrono::steady_clock::now();
        time_span_m = middle-begin;
        i++;
        if(i >= cycles) {
            break;
        }
    }
    output.OutputTriangulation(sys, "int.off");	
    mc_mover.steps_tested_eq = mc_mover.steps_tested_displace + mc_mover.steps_tested_tether + mc_mover.steps_tested_mass + mc_mover.steps_tested_protein + mc_mover.steps_tested_area;
    mc_mover.steps_rejected_eq = mc_mover.steps_rejected_displace + mc_mover.steps_tested_tether + mc_mover.steps_rejected_mass + mc_mover.steps_rejected_protein + mc_mover.steps_rejected_area;
}

void Simulation::Simulate(int cycles, MembraneMC& sys, NeighborList& nl, Analyzers& analyzer, chrono::steady_clock::time_point& begin) {
    // Simulate for number of cycles, or time limit
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    mc_mover.steps_tested_displace = 0;
    mc_mover.steps_rejected_displace = 0;
    mc_mover.steps_tested_tether = 0;
    mc_mover.steps_rejected_tether = 0;
    mc_mover.steps_tested_mass = 0;
    mc_mover.steps_rejected_mass = 0;
    mc_mover.steps_tested_protein = 0;
    mc_mover.steps_rejected_protein = 0;
    mc_mover.steps_tested_area = 0;
    mc_mover.steps_rejected_area = 0;
    ofstream myfile_umb;
    myfile_umb.precision(17);
    myfile_umb.open(sys.output_path+"/mbar_data.txt", std::ios_base::app);
    sys.my_cout.precision(8);
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    int i = 0;
    while(time_span_m.count() < sys.final_warning) {
        util.SaruSeed(sys,sys.count_step);
        sys.count_step++;
	    NextStepParallel(true, sys, nl);
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
            double phi_ = sys.phi;
            double phi_bending_ = sys.phi_bending;
            double phi_phi_ = sys.phi_phi;
            util.InitializeEnergy(sys, nl);
			sys.my_cout << "cycle " << i << endl;
			sys.my_cout << "energy " << std::scientific << sys.phi << " " << std::scientific << sys.phi-phi_ << endl;
            sys.my_cout << "phi_bending " << std::scientific << sys.phi_bending << " " << std::scientific << sys.phi_bending-phi_bending_ << " phi_phi " << std::scientific << sys.phi_phi << " " << std::scientific << sys.phi_phi-phi_phi_ << endl;
            sys.my_cout << "mass " << sys.mass << endl;
            sys.my_cout << "area " << sys.area_total << " and " << sys.lengths[0]*sys.lengths[1] << endl;
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            sys.time_storage_other[0] += time_span.count();

            t1_other = chrono::steady_clock::now();
			util.LinkMaxMin(sys, nl);
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            sys.time_storage_other[1] += time_span.count();

            t1_other = chrono::steady_clock::now();
            sys.my_cout << "displace " << mc_mover.steps_rejected_displace << "/" << mc_mover.steps_tested_displace << endl;
            sys.my_cout << "tether " << mc_mover.steps_rejected_tether << "/" << mc_mover.steps_tested_tether << endl;
            sys.my_cout << "mass " << mc_mover.steps_rejected_mass << "/" << mc_mover.steps_tested_mass << endl;
            sys.my_cout << "protein " << mc_mover.steps_rejected_protein << "/" << mc_mover.steps_tested_protein << endl;
            sys.my_cout << "area " << mc_mover.steps_rejected_area << "/" << mc_mover.steps_tested_area << endl;
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            sys.time_storage_other[2] += time_span.count();
		}
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
			output.OutputTriangulation(sys,"int.off");	
            if(sys.count_step%20000==0) {
                output.OutputTriangulation(sys,"int_2.off");	
            }
            if(i%4000==0) {
			    output.DumpXYZConfig(sys,"config.xyz");
			    output.OutputTriangulationAppend(sys,"prod.off");	
            }
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            sys.time_storage_other[3] += time_span.count();
		}
        if(i%analyzer.storage_umb_time==0) {
            analyzer.UmbOutput(sys.phi, sys.phi_bending, sys.phi_phi, sys.lengths, sys.area_total, myfile_umb);
            analyzer.umb_counts++;
        }
        if(i%analyzer.storage_time==0) {
		    analyzer.energy_storage[i/analyzer.storage_time] = sys.phi;
            analyzer.area_storage[i/analyzer.storage_time] = sys.area_total;
            analyzer.area_proj_storage[i/analyzer.storage_time] = sys.lengths[0]*sys.lengths[1];
            analyzer.mass_storage[i/analyzer.storage_time] = sys.mass;
            analyzer.storage_counts++;
        }
        middle = chrono::steady_clock::now();
        time_span_m = middle-begin;
        i++;
        if(i >= cycles) {
            break;
        }
    }
    output.OutputTriangulation(sys,"int.off");	
    mc_mover.steps_tested_prod = mc_mover.steps_tested_displace + mc_mover.steps_tested_tether + mc_mover.steps_tested_mass + mc_mover.steps_tested_protein + mc_mover.steps_tested_area;
    mc_mover.steps_rejected_prod = mc_mover.steps_rejected_displace + mc_mover.steps_tested_tether + mc_mover.steps_rejected_mass + mc_mover.steps_rejected_protein + mc_mover.steps_rejected_area;
    myfile_umb.close();
}

