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
#include "utilities.hpp"
using namespace std;

Utilities::Utilities() {
    // Constructor
    // Does nothing
}

Utilities::~Utilities() {
    // Destructor
    // Does nothing
}

void Utilities::LinkMaxMin(Membrane& sys, NeighborList& nl) {
    // Find the shortest and longest link
    // Use sys.point_neighbor_list to do the max side
    // Use neighbor lists to do the min side
	double min = pow(10,9);
	double max = -1;
	sys.my_cout << "Link lengths initially" << endl;
    #pragma omp parallel for reduction(max : max) reduction(min : min)
    for(int i=0; i<sys.vertices; i++) {
        for(int j=0; j<sys.point_neighbor_list[i].size(); j++) {
            double link_length = LengthLink(sys,i,sys.point_neighbor_list[i][j]);
            if(link_length > max) {
                max = link_length;
            } 
            if(link_length < min) {
                min = link_length;
            } 
        }
    } 
    #pragma omp parallel for reduction(min : min)
    for(int l=0; l<sys.vertices; l++) {
        int index = nl.neighbor_list_index[l];
	    for(int i=0; i<nl.neighbors[index].size(); i++) {
            for(int j=0; j<nl.neighbor_list[nl.neighbors[index][i]].size(); j++) {
                if(l != nl.neighbor_list[nl.neighbors[index][i]][j]) {
                    double length_neighbor = LengthLink(sys,l,nl.neighbor_list[nl.neighbors[index][i]][j]);
                    if(length_neighbor < min) {
                        min = length_neighbor;
                    } 
                }
            }
        }
    } 
	sys.my_cout << "Min is " << min << "\n";
	sys.my_cout << "Max is " << max << "\n";
}

void Utilities::EnergyNode(Membrane& sys, int i) {
    // Compute energy about a node
	sys.phi_vertex[i] = 0;
    double link_length[sys.point_neighbor_list[i].size()];
    int opposite[sys.point_neighbor_list[i].size()][2];
    double sigma_i = 0;
    double sigma_ij[sys.point_neighbor_list[i].size()];
    double energy_return_x = 0;
    double energy_return_y = 0;
    double energy_return_z = 0;
    // Compute link lengths for neighbor list
    for(int j=0; j<sys.point_neighbor_list[i].size(); j++) {
        int k = sys.point_neighbor_list[i][j];
        link_length[j] = LengthLink(sys,i,k);
        if ((link_length[j] > 1.673) || (link_length[j] < 1.00)) {
            phi_vertex[i] = pow(10,100);
            return;    
        }
    }
    // Find points opposite links
    for(int j_1=0; j_1<sys.point_neighbor_list[i].size(); j_1++) {
        int face_1 = sys.point_neighbor_triangle[i][j_1][0];
        int face_2 = sys.point_neighbor_triangle[i][j_1][1];
        if((sys.triangle_list[face_1][0] != i) && (sys.triangle_list[face_1][0] != point_neighbor_list[i][j_1])) {
            opposite[j_1][0] = sys.triangle_list[face_1][0];
        }
        else if((sys.triangle_list[face_1][1] != i) && (sys.triangle_list[face_1][1] != point_neighbor_list[i][j_1])) {
            opposite[j_1][0] = sys.triangle_list[face_1][1];
        }
        else {
            opposite[j_1][0] = sys.triangle_list[face_1][2];
        }

        if((sys.triangle_list[face_2][0] != i) && (sys.triangle_list[face_2][0] != point_neighbor_list[i][j_1])) {
            opposite[j_1][1] = sys.triangle_list[face_2][0];
        }
        else if((sys.triangle_list[face_2][1] != i) && (sys.triangle_list[face_2][1] != point_neighbor_list[i][j_1])) {
            opposite[j_1][1] = sys.triangle_list[face_2][1];
        }
        else {
            opposite[j_1][1] = sys.triangle_list[face_2][2];
        }
    }

    // Compute sigma_ij
    for(int j=0; j<sys.point_neighbor_list[i].size(); j++) {
        sigma_ij[j] = 0.5*link_length[j]*(Cotangent(sys,opposite[j][0],i,sys.point_neighbor_list[i][j])+Cotangent(sys,opposite[j][1],i,sys.point_neighbor_list[i][j]));
        sigma_i += sigma_ij[j]*link_length[j];
    }
    sigma_i = sigma_i*0.25;
    // Summation over neighbors for energy
    for(int j=0; j<sys.point_neighbor_list[i].size(); j++) {
        double energy_constant = sigma_ij[j]/link_length[j];
        energy_return_x += energy_constant*WrapDistance(sys.radii_tri[i][0], sys.radii_tri[sys.point_neighbor_list[i][j]][0]);
        energy_return_y += energy_constant*WrapDistance(sys.radii_tri[i][1], sys.radii_tri[sys.point_neighbor_list[i][j]][1]);
        energy_return_z += energy_constant*(sys.radii_tri[i][2] - sys.radii_tri[sys.point_neighbor_list[i][j]][2]);
    }
    // Calculate mean curvature
    // Get vertex normal
    // Evaluating vertex normal by taking average weighted by triangle area
    // Energy calculation structured to have area update before this part
    double vertex_normal[3] = {0,0,0};
    double vertex_area = 0;
    for(int j=0; j<sys.point_triangle_list[i].size(); j++) {
        vertex_area += sys.area_faces[sys.point_triangle_list[i][j]];
    }
    for(int j=0; j<sys.point_triangle_list[i].size(); j++) {
        double triangle_normal[3] = {0,0,0};
        NormalTriangle(sys.point_triangle_list[i][j], triangle_normal);
        vertex_normal[0] += sys.area_faces[sys.point_triangle_list[i][j]]/vertex_area*triangle_normal[0];
        vertex_normal[1] += sys.area_faces[sys.point_triangle_list[i][j]]/vertex_area*triangle_normal[1];
        vertex_normal[2] += sys.area_faces[sys.point_triangle_list[i][j]]/vertex_area*triangle_normal[2];
    }
    // Now make sure vector is normalized
    double magnitude = pow(pow(vertex_normal[0],2.0)+pow(vertex_normal[1],2.0)+pow(vertex_normal[2],2.0),0.5);
    vertex_normal[0] = vertex_normal[0]/magnitude;
    vertex_normal[1] = vertex_normal[1]/magnitude;
    vertex_normal[2] = vertex_normal[2]/magnitude;
    // Now can get mean curvature
    sys.mean_curvature_vertex[i] = 1.0/sigma_i*(vertex_normal[0]*energy_return_x+vertex_normal[1]*energy_return_y+vertex_normal[2]*energy_return_z);
    sys.sigma_vertex[i] = sigma_i;
    double diff_curv = sys.mean_curvature_vertex[i]-sys.spon_curv[sys.ising_array[i]];
    sys.phi_vertex[i] = sys.k_b[sys.ising_array[i]]*sigma_i*diff_curv*diff_curv;
}

void Utilities::InitializeEnergy(MembraneMC& sys, NeighborList& nl) {
    sys.phi = 0;
    sys.area_total = 0;
    // Loop through neighbor list to see if any hard sphere constraints are violated
	// Check to make sure not counting self case
    #pragma omp parallel for reduction(+:sys.phi)
    for(int k=0; k<sys.vertices; k++) {
        int index = nl.neighbor_list_index[k];
        for(int i=0; i<nl.neighbors[index].size(); i++) {
            for(int j=0; j<nl.neighbor_list[nl.neighbors[index][i]].size(); j++) {
                // Check particle interactions
                if(k != nl.neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = LengthLink(sys, k,nl.neighbor_list[nl.neighbors[index][i]][j]);
                    if(length_neighbor < 1.0) {
                        sys.phi += pow(10,100);
                    }
                }
            }
        }
    }
    // Compute surface area
    #pragma omp parallel for reduction(+:sys.area_total)
    for(int i=0; i<sys.faces; i++) {
        AreaNode(i);
        sys.area_faces_original[i] = sys.area_faces[i];
        sys.area_total += sys.area_faces[i];
    }
    sys.phi -= sys.tau_frame*sys.lengths[0]*sys.lengths[1];

    sys.phi_bending = 0;
    #pragma omp parallel for reduction(+:sys.phi_bending,sys.phi)
    for(int i=0; i<sys.vertices; i++) {
        EnergyNode(sys, i); 
        sys.phi_vertex_original[i] = sys.phi_vertex[i];
        sys.mean_curvature_vertex_original[i] = sys.mean_curvature_vertex[i];
        sys.sigma_vertex_original[i] = sys.sigma_vertex[i];
        sys.phi += gamma_surf[sys.ising_array[i]]*sys.sigma_vertex[i];
        sys.phi_bending += phi_vertex[i];
    }
    sys.phi += sys.phi_bending;

    // Evaluate Ising model energy
    sys.mass = 0;
    sys.magnet = 0;

    #pragma omp parallel for reduction(+:sys.mass,sys.magnet)
    for(int i=0; i<sys.vertices; i++) {
        if(sys.ising_array[i] < 2) {
            sys.mass += sys.ising_array[i];
            sys.magnet += sys.ising_values[sys.ising_array[i]];
        }
    }
    sys.phi -= sys.h_external*sys.magnet;   
 
    double phi_magnet = 0;
    #pragma omp parallel for reduction(+:phi_magnet)
    for(int i=0; i<sys.vertices; i++) {
        for(int j=0; j<sys.point_neighbor_list[i].size(); j++) {
            phi_magnet -= sys.j_coupling[sys.ising_array[i]][sys.ising_array[sys.point_neighbor_list[i][j]]]*sys.ising_values[sys.ising_array[i]]*sys.ising_values[sys.ising_array[sys.point_neighbor_list[i][j]]];
        }
    }
    sys.phi += 0.5*phi_magnet; // Dividing by 2 as double counting
    sys.phi_phi = 0.5*phi_magnet;
}

void Utilities::InitializeEnergyScale(MembraneMC&, NeighborList& nl) {
    sys.phi = 0;
    sys.area_total = 0;
    // Loop through neighbor list to see if any hard sphere constraints are violated
	// Check to make sure not counting self case
    #pragma omp parallel for reduction(+:sys.phi)
    for(int k=0; k<sys.vertices; k++) {
        int index = nl.neighbor_list_index[k];
        for(int i=0; i<nl.neighbors[index].size(); i++) {
            for(int j=0; j<nl.neighbor_list[nl.neighbors[index][i]].size(); j++) {
                // Check particle interactions
                if(k != nl.neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = LengthLink(sys,k,nl.neighbor_list[nl.neighbors[index][i]][j]);
                    if(length_neighbor < 1.0) {
                        sys.phi += pow(10,100);
                    }
                }
            }
        }
    }
    // If condition violated, Phi > 10^100 so we can just return
    if(sys.phi > pow(10,10)) {
        return;
    }
    // Compute surface area
    #pragma omp parallel for
    for(int i=0; i<sys.faces; i++) {
        AreaNode(sys, i);
    }

    #pragma omp parallel for reduction(+:sys.area_total)
    for(int i=0; i<faces; i++) {
        sys.area_total += sys.area_faces[i];
    }
    sys.phi -= sys.tau_frame*sys.lengths[0]*sys.lengths[1];

    #pragma omp parallel for
    for(int i=0; i<sys.vertices; i++) {
        EnergyNode(sys, i); // Contribution due to mean curvature and surface area
    }

    // Idea is to seperate energy evaluation and adding Phi to allow for sweet vectorization
    sys.phi_bending = 0.0;
    #pragma omp parallel for reduction(+:sys.phi_bending,sys.phi)
    for(int i=0; i<sys.vertices; i++) {
        sys.phi_bending += phi_vertex[i];
        sys.phi += gamma_surf[sys.ising_array[i]]*sys.sigma_vertex[i];
    }
    sys.phi += sys.phi_bending;

    // Evaluate Ising model energy
    sys.phi -= sys.h_external*sys.magnet;   
 
    double phi_magnet = 0;
    #pragma omp parallel for reduction(+:phi_magnet)
    for(int i=0; i<sys.vertices; i++) {
        for(int j=0; j<sys.point_neighbor_list[i].size(); j++) {
            phi_magnet -= sys.j_coupling[sys.ising_array[i]][sys.ising_array[sys.point_neighbor_list[i][j]]]*sys.ising_values[sys.ising_array[i]]*sys.ising_values[sys.ising_array[sys.point_neighbor_list[i][j]]];
        }
    }
    sys.phi += 0.5*phi_magnet; // Dividing by 2 as double counting
    sys.phi_phi = 0.5*phi_magnet;
}

double Utilities::WrapDistance(double a, double b){
    // Performs PBC
    // As working on frame of (-0.5,0,5], just need to round
    double dx = a-b;
    return dx-round(dx);
}

double Utilities::LengthLink(MembraneMC& sys, int i, int j) {
    // Compute distance between two points
    double distX = sys.lengths[0]*WrapDistance(sys.radii_tri[i][0], sys.radii_tri[j][0]);
    double distY = sys.lengths[1]*WrapDistance(sys.radii_tri[i][1], sys.radii_tri[j][1]);
    double distZ = sys.radii_tri[i][2] - sys.radii_tri[j][2];
    return pow(pow(distX,2.0) + pow(distY,2.0) + pow(distZ,2.0),0.5);
}

void Utilities::AreaNode(MembraneMC& sys, int i) {
    // Compute area of a face
    int dummy_1 = sys.triangle_list[i][0];
    int dummy_2 = sys.triangle_list[i][1];
    int dummy_3 = sys.triangle_list[i][2];
    double ac_1 = sys.lengths[0]*WrapDistance(sys.radii_tri[dummy_1][0], sys.radii_tri[dummy_2][0]);
    double ac_2 = sys.lengths[1]*WrapDistance(sys.radii_tri[dummy_1][1], sys.radii_tri[dummy_2][1]);
    double ac_3 = sys.radii_tri[dummy_1][2] - sys.radii_tri[dummy_2][2];
    double bd_1 = sys.lengths[0]*WrapDistance(sys.radii_tri[dummy_1][0], sys.radii_tri[dummy_3][0]);
    double bd_2 = sys.lengths[1]*WrapDistance(sys.radii_tri[dummy_1][1], sys.radii_tri[dummy_3][1]);
    double bd_3 = sys.radii_tri[dummy_1][2] - sys.radii_tri[dummy_3][2];

    // Area is equal to 1/2*magnitude(AB cross AC)
    area_faces[i] = 0.5*pow(pow(ac_2*bd_3-ac_3*bd_2,2.0)+pow(-ac_1*bd_3+ac_3*bd_1,2.0)+pow(ac_1*bd_2-ac_2*bd_1,2.0) , 0.5);
}

void Utilities::NormalTriangle(int i, double normal[3]) {
    // compute normal of a face
    int dummy_1 = sys.triangle_list[i][0];
    int dummy_2 = sys.triangle_list[i][1];
    int dummy_3 = sys.triangle_list[i][2];
    double ac_1 = sys.lengths[0]*WrapDistance(sys.radii_tri[dummy_1][0], sys.radii_tri[dummy_2][0]);
    double ac_2 = sys.lengths[1]*WrapDistance(sys.radii_tri[dummy_1][1], sys.radii_tri[dummy_2][1]);
    double ac_3 = sys.radii_tri[dummy_1][2] - sys.radii_tri[dummy_2][2];
    double bd_1 = sys.lengths[0]*WrapDistance(sys.radii_tri[dummy_1][0], sys.radii_tri[dummy_3][0]);
    double bd_2 = sys.lengths[1]*WrapDistance(sys.radii_tri[dummy_1][1], sys.radii_tri[dummy_3][1]);
    double bd_3 = sys.radii_tri[dummy_1][2] - sys.radii_tri[dummy_3][2];

    // Normal is equal to AC cross BD / magnitude(AC cross BD)
    double magnitude = pow(pow(ac_2*bd_3-ac_3*bd_2,2.0)+pow(-ac_1*bd_3+ac_3*bd_1,2.0)+pow(ac_1*bd_2-ac_2*bd_1,2.0) , 0.5);
    normal[0] = (ac_2*bd_3-ac_3*bd_2)/magnitude;
    normal[1] = (-ac_1*bd_3+ac_3*bd_1)/magnitude;
    normal[2] = (ac_1*bd_2-ac_2*bd_1)/magnitude;
}

void Utilities::ShuffleSaru(Saru& saru, vector<int> &vector_int) {
    for(int i=(vector_int.size()-1); i>0; i--) {
        swap(vector_int[i], vector_int[saru.rand_select(i)]);
    }
}

double Utilities::Cotangent(int i, int j, int k) {
    // Compute angle given by ij, ik
    double ac_1 = sys.lengths[0]*WrapDistance(sys.radii_tri[j][0], sys.radii_tri[i][0]);
    double ac_2 = sys.lengths[1]*WrapDistance(sys.radii_tri[j][1], sys.radii_tri[i][1]);
    double ac_3 = sys.radii_tri[j][2] - sys.radii_tri[i][2];
    double bd_1 = sys.lengths[0]*WrapDistance(sys.radii_tri[k][0], sys.radii_tri[i][0]);
    double bd_2 = sys.lengths[1]*WrapDistance(sys.radii_tri[k][1], sys.radii_tri[i][1]);
    double bd_3 = sys.radii_tri[k][2] - sys.radii_tri[i][2];
    double dot = ac_1*bd_1+ac_2*bd_2+ac_3*bd_3;
    double cross = sqrt(pow(ac_2*bd_3-ac_3*bd_2,2)+pow(-ac_1*bd_3+ac_3*bd_1,2)+pow(ac_1*bd_2-ac_2*bd_1,2));
    return dot/cross;
}
