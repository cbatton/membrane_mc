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

void MembraneMC::InputParam() { // Takes parameters from a file named param
    ifstream input;
    input.open(output_path+"/param");
    cout.rdbuf(myfilebuf);
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        cout << "No input file." << endl;
    }

    else{
        char buffer;
        string line;
        cout << "Param file detected. Changing values." << endl;
        input >> line >> T_list[0] >> T_list[1];
        cout << "T is now " << T_list[0] << " " << T_list[1] << endl;
        getline(input, line);
        input >> line >> k_b[0] >> k_b[1] >> k_b[2];
        cout << "k_b is now " << k_b[0] << " " << k_b[1] << " " << k_b[2] << endl;
        getline(input, line);
        input >> line >> k_g[0] >> k_g[1] >> k_g[2];
        cout << "k_g is now " << k_g[0] << " " << k_g[1] << " " << k_g[2] << endl;
        getline(input, line);
        input >> line >> k_a[0] >> k_a[1] >> k_a[2];
        cout << "k_a is now " << k_a[0] << " " << k_a[1] << " " << k_a[2] << endl;
        getline(input, line);
        input >> line >> gamma_surf[0] >> gamma_surf[1] >> gamma_surf[2];
        cout << "gamma_surf is now " << gamma_surf[0] << " " << gamma_surf[1] << " " << gamma_surf[2] << endl;
        getline(input, line);
        input >> line >> tau_frame;
        cout << "tau_frame is now " << tau_frame << endl;
        getline(input, line);
        input >> line >> spon_curv[0] >> spon_curv[1] >> spon_curv[2]; 
        cout << "Spontaneous curvature is now " << spon_curv[0] << " " << spon_curv[1] << " " << spon_curv[2] << endl;
        getline(input, line);
        input >> line >> Length_x;
        cout << "Length_x is now " << Length_x << endl;
        getline(input, line);
        input >> line >> Length_y;
        cout << "Length_y is now " << Length_y << endl;
        getline(input, line);
        input >> line >> Length_z;
        cout << "Length_z is now " << Length_z << endl;
        getline(input, line);
        input >> line >> lambda;
        cout << "lambda is now " << lambda << endl;
        getline(input, line);
        input >> line >> lambda_scale;
        cout << "lambda_scale is now " << lambda_scale << endl;
        getline(input, line);
        input >> line >> scale;
        cout << "scale is now " << scale << endl;
        getline(input, line);
        input >> line >> ising_values[0] >> ising_values[1] >> ising_values[2];
        cout << "ising_values is now " << ising_values[0] << " " << ising_values[1] << " " << ising_values[2] << endl;
        getline(input, line);
        input >> line >> J_coupling[0][0] >> J_coupling[0][1] >> J_coupling[0][2]; 
        getline(input, line);
        input >> J_coupling[1][0] >> J_coupling[1][1] >> J_coupling[1][2];
        getline(input, line);
        input >> J_coupling[2][0] >> J_coupling[2][1] >> J_coupling[2][2];
        cout << "J is now " << J_coupling[0][0] << " " << J_coupling[0][1] << " " << J_coupling[0][2] << endl;
        cout << "\t" << J_coupling[1][0] << " " << J_coupling[1][1] << " " << J_coupling[1][2] << endl;
        cout << "\t" << J_coupling[2][0] << " " << J_coupling[2][1] << " " << J_coupling[2][2] << endl;
        getline(input, line);
        input >> line >> h_external;
        cout << "h is now " << h_external << endl;
        getline(input, line);
        input >> line >> num_frac;
        cout << "num_frac is now " << num_frac << endl;
        getline(input, line);
        input >> line >> num_proteins;
        cout << "num_proteins is now " << num_proteins << endl;
        getline(input, line);
        input >> line >> seed_base;
        cout << "seed_base is now " << seed_base << endl;
        getline(input, line);
        input >> line >> count_step;
        cout << "count_step is now " << count_step << endl;
        getline(input, line);
        input >> line >> final_time;
        cout << "final_time is now " << final_time << endl;
        getline(input, line);
        input >> line >> nl_move_start;
        cout << "nl_move_start is now " << nl_move_start << endl;
        input.close();
        Length_x_old = Length_x;
        Length_y_old = Length_y;
        Length_z_old = Length_z;
        Length_x_base = Length_x;
        Length_y_base = Length_y;
        // Hash seed_base
        seed_base = seed_base*0x12345677 + 0x12345;
        seed_base = seed_base^(seed_base>>16);
        seed_base = seed_base*0x45679;
        cout << "seed_base is now " << seed_base << endl;
        SaruSeed(count_step);
        final_warning = final_time-60.0;
        spon_curv_end = spon_curv[2];
        spon_curv[2] = 0.0;
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

void MembraneMC::OutputTimes() {
    // Output time analysis
}

void MembraneMC::OutputAnalyzers() {
    // Output analyzers
}
