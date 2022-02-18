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
#include "analyzers.hpp"
#include "utilities.hpp"
#include "saruprng.hpp"
using namespace std;

Analyzers::Analyzers() {
    // Constructor
    // Should need to do nothing here
}

Analyzers::Analyzers(int bins, int storage_time, int storage_umb_time, MembraneMC& sys) : bins(bins), storage_time(storage_time), storage_umb_time(storage_umb_time) {
    // Adjust size of storage variables
    int storage = sys.cycles_prod/storage_time;
    energy_storage.resize(storage,0.0);
    area_storage.resize(storage,0.0);
    area_proj_storage.resize(storage,0.0);
    mass_storage.resize(storage,0.0);
    // Adjust rho variables
    vector<double> list(bins,0.0);
    rho.resize(6,list);
    bin_size = sys.lengths[0]/bins;
    // Adjust output_path
    output_path = sys.output_path;
    
}

Analyzers::~Analyzers() {
    // Destructor
    // Should need to do nothing here
}

void Analyzers::EnergyAnalyzer() {
    // Evaluate average
    double energy_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        energy_ave += energy_storage[i];
    }
    energy_ave = energy_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double energy_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        energy_std += pow(energy_ave-energy_storage[i],2);
    }
    energy_std = sqrt(energy_std/(storage_counts-1));

    ofstream myfile;
    myfile.open(output_path+"/energy.txt", std::ios_base::app);
    myfile << "Energy from simulation run" << endl;
    myfile << "Average " << std::scientific << energy_ave << " Standard deviation " << std::scientific << energy_std << endl;
    myfile.close();

    myfile.open(output_path+"/energy_storage.txt", std::ios_base::app);
    myfile << "Energy from run" << endl;
    myfile.precision(8);
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << energy_storage[i] << endl;
    }
    myfile.close();
}

void Analyzers::AreaAnalyzer() {
    // Evaluate average
    double area_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_ave += area_storage[i];
    }
    area_ave = area_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double area_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_std += pow(area_ave-area_storage[i],2);
    }
    area_std = sqrt(area_std/(storage_counts-1));

    ofstream myfile;
    myfile.open(output_path+"/area.txt", std::ios_base::app);
    myfile.precision(17);
    myfile << "Area from simulation run" << endl;
    myfile << "Average " << std::scientific << area_ave << " Standard deviation " << std::scientific << area_std << endl;
    myfile.close();

    myfile.open(output_path+"/area_storage.txt", std::ios_base::app);
    myfile.precision(8);
    myfile << "Area from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << area_storage[i] << endl;
    }
    myfile.close();
}

void Analyzers::AreaProjAnalyzer() {
    // Evaluate average
    double area_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_ave += area_proj_storage[i];
    }
    area_ave = area_ave/storage_counts;
    area_proj_average = area_ave;
    // Evaluate standard deviation using Bessel's correction
    double area_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_std += pow(area_ave-area_proj_storage[i],2);
    }
    area_std = sqrt(area_std/(storage_counts-1));

    ofstream myfile;
    myfile.open(output_path+"/area_proj.txt", std::ios_base::app);
    myfile.precision(17);
    myfile << "Area from simulation run" << endl;
    myfile << "Average " << std::scientific << area_ave << " Standard deviation " << std::scientific << area_std << endl;
    myfile.close();

    myfile.open(output_path+"/area_proj_storage.txt", std::ios_base::app);
    myfile.precision(8);
    myfile << "Area from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << area_proj_storage[i] << endl;
    }
    myfile.close();
}

void Analyzers::MassAnalyzer() {
    // Evaluate average
    double mass_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        mass_ave += mass_storage[i];
    }
    mass_ave = mass_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double mass_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        mass_std += pow(mass_ave-mass_storage[i],2);
    }
    mass_std = sqrt(mass_std/(storage_counts-1));

    ofstream myfile;
    myfile.open(output_path+"/mass.txt", std::ios_base::app);
    myfile.precision(17);
    myfile << "Mass from simulation run" << endl;
    myfile << "Average " << std::scientific << mass_ave << " Standard deviation " << std::scientific << mass_std << endl;
    myfile.close();

    myfile.open(output_path+"/mass_storage.txt", std::ios_base::app);
    myfile.precision(8);
    myfile << "Mass from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << mass_storage[i] << endl;
    }
    myfile.close();
}

void Analyzers::UmbOutput(double& phi, double& phi_bending, double& phi_phi, vector<double>& lengths, double& area_total, ofstream& myfile) {
    // Output current umbrella variables
    // Evaluate average of energy to scale results
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    myfile << std::scientific << " " << std::scientific << phi << " " << std::scientific << phi_bending << " " << std::scientific << phi_phi << " " << std::scientific << lengths[0]*lengths[1] << " " << std::scientific << area_total << "\n";
}

void Analyzers::ClusterAnalysis(MembraneMC& sys) {
    // Do cluster analysis on current lattice configuration
    // Will find all clusters that have proteins/species associated with protein
    // Iterate over all the proteins as a start, find clusters from there
    // Note that clusters with just species associated with protein are not counted
    //
    // Initialize cluster list
    cluster cluster_cur;
    cluster_cur.vertex_status.resize(sys.vertices, -1);
    cluster_cur.cluster_list.clear();
    // Begin iteration    
    vector<int> list;
    for(int i=0; i<sys.vertices; i++) {
        if(sys.protein_node[i] == 0) {
            if(cluster_cur.vertex_status[i] == -1) {
                cluster_cur.cluster_list.push_back(list);
                ClusterDFS(sys, i, cluster_cur.cluster_list.size()-1, cluster_cur);
            }
        }
    }

    // Output current cluster statistics
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.open(output_path+"/cluster.txt", std::ios_base::app);
    myfile << "# " << sys.count_step << endl;
    for(int i=0; i<cluster_cur.cluster_list.size(); i++) {
        int protein_in_cluster = 0;
        for(int j=0; j<cluster_cur.cluster_list[i].size(); j++) {
            if(sys.ising_array[cluster_cur.cluster_list[i][j]] == 2) {
                protein_in_cluster++;
            }
        }
        myfile << cluster_cur.cluster_list[i].size() << " " << protein_in_cluster << "\n";
    }
    //Do analysis to append to counting vectors
    int number_clusters_ = cluster_cur.cluster_list.size();
    double mean_cluster_number_ = 0;
    double mean_cluster_weight_ = 0;
    double mean_cluster_number_protein_ = 0;
    double mean_cluster_weight_protein_ = 0;
    for(int i=0; i<cluster_cur.cluster_list.size(); i++) {
        int cluster_size_ = cluster_cur.cluster_list[i].size();
        int cluster_size_protein_ = 0;
        for(int j=0; j<cluster_cur.cluster_list[i].size(); j++) {
            if(sys.ising_array[cluster_cur.cluster_list[i][j]] == 2) {
                cluster_size_protein_++;
            }
        }
        mean_cluster_number_ += cluster_size_;
        mean_cluster_number_protein_ += cluster_size_protein_;
        mean_cluster_weight_ += cluster_size_*cluster_size_;
        mean_cluster_weight_protein_ += cluster_size_protein_*cluster_size_protein_;
    }
    mean_cluster_weight_ /= mean_cluster_number_;
    mean_cluster_weight_protein_ /= mean_cluster_number_protein_;
    mean_cluster_number_ /= number_clusters_;
    mean_cluster_number_protein_ /= number_clusters_;
    //Push back to storage
    number_clusters.push_back(number_clusters_);
    mean_cluster_number.push_back(mean_cluster_number_);
    mean_cluster_weight.push_back(mean_cluster_weight_);
    mean_cluster_number_protein.push_back(mean_cluster_number_protein_);
    mean_cluster_weight_protein.push_back(mean_cluster_weight_protein_);
}

void Analyzers::ClusterDFS(MembraneMC& sys, int vertex_ind, int cluster_ind, cluster& cluster_cur) {
    // do recursive depth-first search
    cluster_cur.vertex_status[vertex_ind] = 0;
    cluster_cur.cluster_list[cluster_ind].push_back(vertex_ind);
    for(int i=0; i<sys.point_neighbor_list[vertex_ind].size(); i++) {
        int neighbor = sys.point_neighbor_list[vertex_ind][i];
        if(cluster_cur.vertex_status[neighbor] == -1) {
            if(sys.ising_array[neighbor] != 0) {
                ClusterDFS(sys, neighbor, cluster_ind, cluster_cur);
            }
        }
    }
}

void Analyzers::ClusterPostAnalysis() {
    //Do analysis on the cluster vectors
    // Evaluate averages
    double mean_cluster_number_ave = 0.0;
    double mean_cluster_weight_ave = 0.0;
    double mean_cluster_number_protein_ave = 0.0;
    double mean_cluster_weight_protein_ave = 0.0;
    double number_clusters_ave = 0.0;
    int storage_counts = mean_cluster_number.size();
    for(int i=0; i<storage_counts; i++) {
        mean_cluster_number_ave += mean_cluster_number[i];
        mean_cluster_weight_ave += mean_cluster_weight[i];
        mean_cluster_number_protein_ave += mean_cluster_number_protein[i];
        mean_cluster_weight_protein_ave += mean_cluster_weight_protein[i];
        number_clusters_ave += number_clusters[i];
    }
    mean_cluster_number_ave = mean_cluster_number_ave/storage_counts;
    mean_cluster_weight_ave = mean_cluster_weight_ave/storage_counts;
    mean_cluster_number_protein_ave = mean_cluster_number_protein_ave/storage_counts;
    mean_cluster_weight_protein_ave = mean_cluster_weight_protein_ave/storage_counts;
    number_clusters_ave = number_clusters_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double mean_cluster_number_std = 0.0;
    double mean_cluster_weight_std = 0.0;
    double mean_cluster_number_protein_std = 0.0;
    double mean_cluster_weight_protein_std = 0.0;
    double number_clusters_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        mean_cluster_number_std += pow(mean_cluster_number_ave-mean_cluster_number[i],2);
        mean_cluster_weight_std += pow(mean_cluster_weight_ave-mean_cluster_weight[i],2);
        mean_cluster_number_protein_std += pow(mean_cluster_number_protein_ave-mean_cluster_number_protein[i],2);
        mean_cluster_weight_protein_std += pow(mean_cluster_weight_protein_ave-mean_cluster_weight_protein[i],2);
        number_clusters_std += pow(number_clusters_ave-number_clusters[i],2);
    }
    mean_cluster_number_std = sqrt(mean_cluster_number_std/(storage_counts-1));
    mean_cluster_weight_std = sqrt(mean_cluster_weight_std/(storage_counts-1));
    mean_cluster_number_protein_std = sqrt(mean_cluster_number_protein_std/(storage_counts-1));
    mean_cluster_weight_protein_std = sqrt(mean_cluster_weight_protein_std/(storage_counts-1));
    number_clusters_std = sqrt(number_clusters_std/(storage_counts-1));

    ofstream myfile;
    myfile.precision(10);
    myfile.open(output_path+"/cluster_analysis.txt", std::ios_base::app);
    myfile << "# Cluster analysis from simulation run" << endl;
    myfile << "MCN " << std::scientific << mean_cluster_number_ave << " " << std::scientific << mean_cluster_number_std << endl;
    myfile << "MCW " << std::scientific << mean_cluster_weight_ave << " " << std::scientific << mean_cluster_weight_std << endl;
    myfile << "MCPN " << std::scientific << mean_cluster_number_protein_ave << " " << std::scientific << mean_cluster_number_protein_std << endl;
    myfile << "MCPW " << std::scientific << mean_cluster_weight_protein_ave << " " << std::scientific << mean_cluster_weight_protein_std << endl;
    myfile << "NC " << std::scientific << number_clusters_ave << " " << std::scientific << number_clusters_std << endl;
    myfile.close();

    myfile.open(output_path+"/cluster_analysis_storage.txt", std::ios_base::app);
    myfile << "# Cluster analysis from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << mean_cluster_number[i] << " " << std::scientific << mean_cluster_weight[i] << " " << std::scientific << mean_cluster_number_protein[i] << " " << std::scientific << mean_cluster_weight_protein[i] << " " << number_clusters[i] << endl;
    }
    myfile.close();
}

void Analyzers::RDFRoutine(MembraneMC& sys, int ind, int spec_0, int spec_1, int spec_2, int spec_3) {
    // routine to sample rho
    Utilities util;
    #pragma omp parallel for schedule(dynamic,32)
    for(int i=0; i<sys.vertices; i++) {
        if((sys.ising_array[i] == spec_0) || (sys.ising_array[i] == spec_1)) {
            for(int j=0; j<sys.vertices; j++) {
                if(i != j) {
                    if((sys.ising_array[j] == spec_2) || (sys.ising_array[j] == spec_3)) {
                        double distance = util.LengthLink(sys,i,j);
                        int bin_loc = int(distance/bin_size);
                        if((distance<sys.lengths[0]*0.5) && (bin_loc < bins)) {
                            #pragma omp atomic
                            rho[ind][bin_loc] += 1;
                        }
                    }
                }
            }
        }
    }
}

void Analyzers::RhoSample(MembraneMC& sys) {
    // Sample radial distribution function
    // Will be measuring a bunch of these...
    int pairs[6][4];
    // 2-2
    pairs[0][0] = 2;
    pairs[0][1] = 2;
    pairs[0][2] = 2;
    pairs[0][3] = 2;
    // 2-1 
    pairs[1][0] = 2;
    pairs[1][1] = 2;
    pairs[1][2] = 1;
    pairs[1][3] = 1;
    // 2-0 
    pairs[2][0] = 2;
    pairs[2][1] = 2;
    pairs[2][2] = 0;
    pairs[2][3] = 0;
    // 1-1
    pairs[3][0] = 1;
    pairs[3][1] = 1;
    pairs[3][2] = 1;
    pairs[3][3] = 1;
    // 1-0
    pairs[4][0] = 1;
    pairs[4][1] = 1;
    pairs[4][2] = 0;
    pairs[4][3] = 0;
    // 0-0
    pairs[5][0] = 0;
    pairs[5][1] = 0;
    pairs[5][2] = 0;
    pairs[5][3] = 0;
    for(int i=0; i<6; i++) {
        RDFRoutine(sys, i, pairs[i][0], pairs[i][1], pairs[i][2], pairs[i][3]);
    }
    rdf_sample++;
    mass_sample[0] += sys.vertices-sys.num_proteins-sys.mass;
    mass_sample[1] += sys.mass;
    mass_sample[2] += sys.num_proteins;

}

void Analyzers::RhoAnalyzer(MembraneMC& sys) {
    // Analyze the mean density with respect to the protein center
    mass_sample[0] /= rdf_sample;
    mass_sample[1] /= rdf_sample;
    mass_sample[2] /= rdf_sample;
    double area_change = 0;
    double nIdeal = 0;
    // Evaluate average projected area
    double area_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_ave += area_proj_storage[i];
    }
    area_ave = area_ave/storage_counts;
    area_proj_average = area_ave;
    double rho_ideal[6];
    rho_ideal[0] = mass_sample[2]/(area_proj_average);
    rho_ideal[1] = mass_sample[2]/(area_proj_average);
    rho_ideal[2] = mass_sample[2]/(area_proj_average);
    rho_ideal[3] = mass_sample[1]/(area_proj_average);
    rho_ideal[4] = mass_sample[1]/(area_proj_average);
    rho_ideal[5] = mass_sample[0]/(area_proj_average);
    double num_count[6];
    num_count[0] = mass_sample[2];
    num_count[1] = mass_sample[1];
    num_count[2] = mass_sample[0];
    num_count[3] = mass_sample[1];
    num_count[4] = mass_sample[0];
    num_count[5] = mass_sample[0];
    for(int k=0; k<6; k++) {
        #pragma omp parallel for
        for(int i=0; i<bins; i++) {
            area_change = (pow(i+1,2)-pow(i,2))*pow(bin_size,2);
            nIdeal = M_PI*area_change*rho_ideal[k];
            rho[k][i] = rho[k][i]/(nIdeal*num_count[k]*rdf_sample);
        }
    }
    // Output to file
    for(int k=0; k<6; k++) {
        ofstream myfile;
        myfile.precision(10);
        myfile.open(output_path+"/rho_protein_"+to_string(k)+".txt", std::ios_base::app);
        myfile << "area_proj_average " << area_proj_average << endl;
        myfile << "Length_x*Length_y " << sys.lengths[0]*sys.lengths[1] << endl;
        myfile << "Bin size " << bin_size << endl;
        for(int i=0; i<bins; i++) {
            myfile << bin_size*(i+0.5) << " " << std::scientific << rho[k][i] << endl;
        }
        myfile.close();
    }
    // Output masses for post-processing
    ofstream myfile;
    myfile.precision(10);
    myfile.open(output_path+"/rho_protein_mass.txt", std::ios_base::app);
    myfile << std::scientific << mass_sample[0] << " " << std::scientific << mass_sample[1] << " " << std::scientific << mass_sample[2] << endl;
}

void Analyzers::OutputAnalyzers(MembraneMC& sys) {
    // Output analyzers
    sys.t1 = chrono::steady_clock::now();
    EnergyAnalyzer();
    AreaAnalyzer();
    AreaProjAnalyzer();
    MassAnalyzer();
    ClusterPostAnalysis();
    RhoAnalyzer(sys);
    sys.t2 = chrono::steady_clock::now();
    chrono::duration<double> time_span_anal = sys.t2-sys.t1;
    sys.time_storage_other[5] += time_span_anal.count();
}
