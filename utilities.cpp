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

void Utilities::LinkMaxMin() {
    cout.rdbuf(myfilebuf);
	double min = pow(10,9);
	double max = -1;
	cout << "Link lengths initially" << endl;
    #pragma omp parallel for reduction(max : max) reduction(min : min)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            double link_length = lengthLink(i,point_neighbor_list[i][j]);
            // cout << i << " " << j << " " << link_length << endl;
            /*
            if (link_length < 1.4) {
                cout << "Too low!" << endl;
                cout << "i " << Radius_x_tri[i] << " " << Radius_y_tri[i] << " " << Radius_z_tri[i] << endl;
                cout << "j " << Radius_x_tri[j] << " " << Radius_y_tri[j] << " " << Radius_z_tri[j] << endl;
            }
            if (link_length > 1.6) {
                cout << "Too high!" << endl;
                cout << "i " << Radius_x_tri[i] << " " << Radius_y_tri[i] << " " << Radius_z_tri[i] << endl;
                cout << "j " << Radius_x_tri[j] << " " << Radius_y_tri[j] << " " << Radius_z_tri[j] << endl;
            }
            */
            if(link_length > max) {
                max = link_length;
            } 
            if(link_length < min) {
                min = link_length;
            } 
        }
    } 
    // Let's change this to using the neighbor list.......
    #pragma omp parallel for reduction(min : min)
    for(int l=0; l<vertices; l++) {
        int index = neighbor_list_index[l];
	    for(int i=0; i<neighbors[index].size(); i++) {
            for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
                if(l != neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = lengthLink(l,neighbor_list[neighbors[index][i]][j]);
                    if(length_neighbor < min) {
                        min = length_neighbor;
                    } 
                }
            }
        }
    } 
	cout << "Min is " << min << "\n";
	cout << "Max is " << max << "\n";
}

void Utilities::EnergyNode(int i) {
    // Compute energy about a node
	phi_vertex[i] = 0;
    double link_length[point_neighbor_max[i]];
    double opposite_angles[point_neighbor_max[i]][2];
    double sigma_i = 0;
    double sigma_ij[point_neighbor_max[i]];
    double energy_return_x = 0;
    double energy_return_y = 0;
    double energy_return_z = 0;
    // Compute link lengths for neighbor list
    for(int j=0; j<point_neighbor_max[i]; j++) {
        int k = point_neighbor_list[i][j];
        link_length[j] = lengthLink(i,k);
        if ((link_length[j] > 1.673) || (link_length[j] < 1.00)) {
            phi_vertex[i] = pow(10,100);
			// cout << "Breaks constraints at " << i << " " << k << endl;
			// cout << link_length[j] << endl;
            return;    
        }
        // cout << "Link length at " << i << " " << k << " " << " is " << link_length[j] << endl;
    }
    // Compute angles opposite links
    for(int j_1=0; j_1<point_neighbor_max[i]; j_1++) {
        int face_1 = point_neighbor_triangle[i][j_1][0];
        int face_2 = point_neighbor_triangle[i][j_1][1];
        // cout << "Points are for triangle are " << i << " " << point_neighbor_list[i][j_1] << endl;
        // cout << "Triangle 1 " << triangle_list[face_1][0] << " " << triangle_list[face_1][1] << " " << triangle_list[face_1][2] << endl;
        // cout << "Triangle 2 " << triangle_list[face_2][0] << " " << triangle_list[face_2][1] << " " << triangle_list[face_2][2] << endl;
        // From given faces, determine point in triangles not given by i or neighbor list 
        // Then use cosineAngle to get the angles, with opposite being the location opposite
        if((triangle_list[face_1][0] != i) && (triangle_list[face_1][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][0];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_1][1] != i) && (triangle_list[face_1][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][1];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_1][2];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }

        if((triangle_list[face_2][0] != i) && (triangle_list[face_2][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][0];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_2][1] != i) && (triangle_list[face_2][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][1];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_2][2];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
		/*
        if((opposite_angles[j_1][1] > M_PI_2*1.18) || (opposite_angles[j_1][0] > M_PI_2*1.18)) {
           phi_vertex[i] = 1000000;
           return;
        }
		*/
    }

    // Compute sigma_ij
    for(int j=0; j<point_neighbor_max[i]; j++) {
        // cout << "Opposite angles at " << i << " are " << opposite_angles[j][0] << " and " << opposite_angles[j][1] << endl;
        sigma_ij[j] = link_length[j]*(cotangent_fast(opposite_angles[j][0])+cotangent_fast(opposite_angles[j][1]))*0.5;
        // cout << "Sigma_ij at " << j << " is " << sigma_ij[j] << endl;
        sigma_i += sigma_ij[j]*link_length[j];
    }
    sigma_i = sigma_i*0.25;
    // sigma_i_total += sigma_i;
    // cout << "sigma_i is " << sigma_i << endl;
    // Summation over neighbors for energy
    // cout << "Initial energy_return_z " << energy_return_z << endl;
    for(int j=0; j<point_neighbor_max[i]; j++) {
        double energy_constant = sigma_ij[j]/link_length[j];
        // cout << "Energy constant is " << energy_constant << endl;
        energy_return_x += energy_constant*wrapDistance_x(Length_x*Radius_x_tri[i], Length_x*Radius_x_tri[point_neighbor_list[i][j]]);
        energy_return_y += energy_constant*wrapDistance_y(Length_y*Radius_y_tri[i], Length_y*Radius_y_tri[point_neighbor_list[i][j]]);
        energy_return_z += energy_constant*(Radius_z_tri[i] - Radius_z_tri[point_neighbor_list[i][j]]);
        // cout << "Z different is " << Radius_z_tri[i] << " minus " << Radius_z_tri[point_neighbor_list[i][j]] << " equals " << energy_return_z << endl;
    }
    // cout << "Energy values are x: " << energy_return_x << " y: " << energy_return_y << " z: " << energy_return_z << endl;
    // Don't ask me why but this comment makes the values of phi stable
    // I'm serious
    // cout << "Energy at vertex " << i << " : " << phi_vertex[i] << endl;
    // Calculate mean curvature if a protein node
    // get vertex normal
    // Evaluating vertex normal by taking average weighted by triangle area
    // Energy calculation structured to have area update before this part
    double vertex_normal[3] = {0,0,0};
    double vertex_area = 0;
    for(int j=0; j<point_triangle_max[i]; j++) {
        vertex_area += area_faces[point_triangle_list[i][j]];
    }
    for(int j=0; j<point_triangle_max[i]; j++) {
        double triangle_normal[3] = {0,0,0};
        normalTriangle(point_triangle_list[i][j], triangle_normal);
        vertex_normal[0] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[0];
        vertex_normal[1] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[1];
        vertex_normal[2] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[2];
    }
    // Now make sure vector is normalized
    double magnitude = pow(pow(vertex_normal[0],2.0)+pow(vertex_normal[1],2.0)+pow(vertex_normal[2],2.0),0.5);
    vertex_normal[0] = vertex_normal[0]/magnitude;
    vertex_normal[1] = vertex_normal[1]/magnitude;
    vertex_normal[2] = vertex_normal[2]/magnitude;
    // Now can get mean curvature
    mean_curvature_vertex[i] = 1.0/sigma_i*(vertex_normal[0]*energy_return_x+vertex_normal[1]*energy_return_y+vertex_normal[2]*energy_return_z);
    sigma_vertex[i] = sigma_i;
    double diff_curv = mean_curvature_vertex[i]-spon_curv[Ising_Array[i]];
    phi_vertex[i] = k_b[Ising_Array[i]]*sigma_i*diff_curv*diff_curv;
}

void Utilities::InitializeEnergy() {
    Phi = 0;
    Area_total = 0;
    // Loop through neighbor list to see if any hard sphere constraints are violated
	// Check to make sure not counting self case
    #pragma omp parallel for reduction(+:Phi)
    for(int k=0; k<vertices; k++) {
        int index = neighbor_list_index[k];
        for(int i=0; i<neighbors[index].size(); i++) {
            for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
                // Check particle interactions
                if(k != neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = lengthLink(k,neighbor_list[neighbors[index][i]][j]);
                    if(length_neighbor < 1.0) {
                        Phi += pow(10,100);
                    }
                }
            }
        }
    }
    // Compute surface area
    #pragma omp parallel for reduction(+:Area_total,Phi)
    for(int i=0; i<faces; i++) {
        areaNode(i);
        area_faces_original[i] = area_faces[i];
        Area_total += area_faces[i];
        Phi += gamma_surf[0]*area_faces[i]; // On second though, gamma_surf here might be as easy as assigning a value from Ising_Array as it is associated with a face value rather than a vertix. Maybe best to take average of vertices on face
    }
    Phi -= tau_frame*Length_x*Length_y;

    Phi_bending = 0;
    #pragma omp parallel for reduction(+:Phi_bending)
    for(int i=0; i<vertices; i++) {
        energyNode(i); // Contribution due to mean curvature and surface area
        phi_vertex_original[i] = phi_vertex[i];
        mean_curvature_vertex_original[i] = mean_curvature_vertex[i];
        sigma_vertex_original[i] = sigma_vertex[i];
        Phi_bending += phi_vertex[i];
    }
    Phi += Phi_bending;

    // Evaluate Ising model like energy
    Mass = 0;
    Magnet = 0;

    #pragma omp parallel for reduction(+:Mass,Magnet)
    for(int i=0; i<vertices; i++) {
        if(Ising_Array[i] < 2) {
            Mass += Ising_Array[i];
            Magnet += ising_values[Ising_Array[i]];
        }
    }
    Phi -= h_external*Magnet;   
 
    double Phi_magnet = 0;
    #pragma omp parallel for reduction(+:Phi_magnet)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            Phi_magnet -= J_coupling[Ising_Array[i]][Ising_Array[point_neighbor_list[i][j]]]*ising_values[Ising_Array[i]]*ising_values[Ising_Array[point_neighbor_list[i][j]]];
        }
    }
    Phi += Phi_magnet*0.5; // Dividing by 2 as double counting
}

void Utilities::InitializeEnergyScale() {
    Phi = 0;
    Area_total = 0;
    // Loop through neighbor list to see if any hard sphere constraints are violated
	// Check to make sure not counting self case
    #pragma omp parallel for reduction(+:Phi)
    for(int k=0; k<vertices; k++) {
        int index = neighbor_list_index[k];
        for(int i=0; i<neighbors[index].size(); i++) {
            for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
                // Check particle interactions
                if(k != neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = lengthLink(k,neighbor_list[neighbors[index][i]][j]);
                    if(length_neighbor < 1.0) {
                        Phi += pow(10,100);
                    }
                }
            }
        }
    }

    // If condition violated, Phi > 10^100 so we can just return
    if(Phi > pow(10,10)) {
        return;
    }
    // Compute surface area
    #pragma omp parallel for
    for(int i=0; i<faces; i++) {
        areaNode(i);
    }

    #pragma omp parallel for reduction(+:Area_total,Phi)
    for(int i=0; i<faces; i++) {
        Area_total += area_faces[i];
        Phi += gamma_surf[0]*area_faces[i]; // On second though, gamma_surf here might be as easy as assigning a value from Ising_Array as it is associated with a face value rather than a vertix. Maybe best to take average of vertices on face
    }
    Phi -= tau_frame*Length_x*Length_y;

    #pragma omp parallel for
    for(int i=0; i<vertices; i++) {
        energyNode(i); // Contribution due to mean curvature and surface area
    }

    // Idea is to seperate energy evaluation and adding Phi to allow for sweet vectorization
    Phi_bending = 0.0;
    #pragma omp parallel for reduction(+:Phi_bending)
    for(int i=0; i<vertices; i++) {
        Phi_bending += phi_vertex[i];
    }
    Phi += Phi_bending;

    // Evaluate Ising model like energy
    Phi -= h_external*Magnet;   
 
    double Phi_magnet = 0;
    #pragma omp parallel for reduction(+:Phi_magnet)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            Phi_magnet -= J_coupling[Ising_Array[i]][Ising_Array[point_neighbor_list[i][j]]]*ising_values[Ising_Array[i]]*ising_values[Ising_Array[point_neighbor_list[i][j]]];
        }
    }
    Phi += Phi_magnet*0.5; // Dividing by 2 as double counting
    Phi_phi = Phi_magnet*0.5;
}

void Utilities::EnergyNode_i(int i) {
    // Compute energy about a node
	phi_vertex[i] = 0;
    double link_length[point_neighbor_max[i]];
    double opposite_angles[point_neighbor_max[i]][2];
    double sigma_i = 0;
    double sigma_ij[point_neighbor_max[i]];
    double energy_return_x = 0;
    double energy_return_y = 0;
    double energy_return_z = 0;
    // Compute link lengths for neighbor list
    for(int j=0; j<point_neighbor_max[i]; j++) {
        int k = point_neighbor_list[i][j];
        link_length[j] = lengthLink(i,k);
        if ((link_length[j] > 1.673) || (link_length[j] < 1.00)) {
            phi_vertex[i] = pow(10,100);
			// cout << "Breaks constraints at " << i << " " << k << endl;
            // return;    
        }
        // cout << "Link length at " << i << " " << k << " " << " is " << link_length[j] << endl;
    }
    // Compute angles opposite links
    for(int j_1=0; j_1<point_neighbor_max[i]; j_1++) {
        int face_1 = 0;
        int face_2 = 0;
        face_1 = point_neighbor_triangle[i][j_1][0];
        face_2 = point_neighbor_triangle[i][j_1][1];
        // cout << "Points are for triangle are " << i << " " << point_neighbor_list[i][j_1] << endl;
        // cout << "Triangle 1 " << triangle_list[face_1][0] << " " << triangle_list[face_1][1] << " " << triangle_list[face_1][2] << endl;
        // cout << "Triangle 2 " << triangle_list[face_2][0] << " " << triangle_list[face_2][1] << " " << triangle_list[face_2][2] << endl;
        // From given faces, determine point in triangles not given by i or neighbor list 
        // Then use cosineAngle to get the angles, with opposite being the location opposite
        if((triangle_list[face_1][0] != i) && (triangle_list[face_1][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][0];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_1][1] != i) && (triangle_list[face_1][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][1];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_1][2];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }

        if((triangle_list[face_2][0] != i) && (triangle_list[face_2][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][0];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_2][1] != i) && (triangle_list[face_2][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][1];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_2][2];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
		/*
        if((opposite_angles[j_1][1] > M_PI_2*1.18) || (opposite_angles[j_1][0] > M_PI_2*1.18)) {
           phi_vertex[i] = 1000000;
           return;
        }
		*/
    }

    // Compute sigma_ij
    for(int j=0; j<point_neighbor_max[i]; j++) {
        // cout << "Opposite angles at " << i << " are " << opposite_angles[j][0] << " and " << opposite_angles[j][1] << endl;
        sigma_ij[j] = link_length[j]*(cotangent_fast(opposite_angles[j][0])+cotangent_fast(opposite_angles[j][1]))/2.0;
        // cout << "Sigma_ij at " << j << " is " << sigma_ij[j] << endl;
        sigma_i += sigma_ij[j]*link_length[j];
    }
    sigma_i = sigma_i/4.0;
    #pragma omp atomic
    sigma_i_total += sigma_i;
    // cout << "sigma_i is " << sigma_i << endl;
    // Summation over neighbors for energy
    // cout << "Initial energy_return_z " << energy_return_z << endl;
    for(int j=0; j<point_neighbor_max[i]; j++) {
        double energy_constant = sigma_ij[j]/link_length[j];
        // cout << "Energy constant is " << energy_constant << endl;
        energy_return_x += energy_constant*wrapDistance_x(Length_x*Radius_x_tri[i], Length_x*Radius_x_tri[point_neighbor_list[i][j]]);
        energy_return_y += energy_constant*wrapDistance_y(Length_y*Radius_y_tri[i], Length_y*Radius_y_tri[point_neighbor_list[i][j]]);
        energy_return_z += energy_constant*(Radius_z_tri[i] - Radius_z_tri[point_neighbor_list[i][j]]);
        // cout << "Z different is " << Radius_z_tri[i] << " minus " << Radius_z_tri[point_neighbor_list[i][j]] << " equals " << energy_return_z << endl;
    }
    // cout << "Energy values are x: " << energy_return_x << " y: " << energy_return_y << " z: " << energy_return_z << endl;
    double vertex_normal[3] = {0,0,0};
    double vertex_area = 0;
    for(int j=0; j<point_triangle_max[i]; j++) {
        vertex_area += area_faces[point_triangle_list[i][j]];
    }
    for(int j=0; j<point_triangle_max[i]; j++) {
        double triangle_normal[3] = {0,0,0};
        normalTriangle(point_triangle_list[i][j], triangle_normal);
        vertex_normal[0] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[0];
        vertex_normal[1] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[1];
        vertex_normal[2] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[2];
    }
    // Now make sure vector is normalized
    double magnitude = pow(pow(vertex_normal[0],2.0)+pow(vertex_normal[1],2.0)+pow(vertex_normal[2],2.0),0.5);
    vertex_normal[0] = vertex_normal[0]/magnitude;
    vertex_normal[1] = vertex_normal[1]/magnitude;
    vertex_normal[2] = vertex_normal[2]/magnitude;
    // Now can get mean curvature
    mean_curvature_vertex[i] = 1.0/sigma_i*(vertex_normal[0]*energy_return_x+vertex_normal[1]*energy_return_y+vertex_normal[2]*energy_return_z);
    sigma_vertex[i] = sigma_i;
    double diff_curv = mean_curvature_vertex[i]-spon_curv[Ising_Array[i]];
    phi_vertex[i] = k_b[Ising_Array[i]]*sigma_i*diff_curv*diff_curv;
}


void Utilities::InitializeEnergy_i() {
    Phi = 0;
    Area_total = 0;
    // Compute surface area
    #pragma omp parallel for reduction(+:Area_total,Phi)
    for(int i=0; i<faces; i++) {
        areaNode(i);
        area_faces_original[i] = area_faces[i];
        Area_total += area_faces[i];
        Phi += gamma_surf[0]*area_faces[i]; // On second though, gamma_surf here might be as easy as assigning a value from Ising_Array as it is associated with a face value rather than a vertix. Maybe best to take average of vertices on face
    }
    Phi -= tau_frame*Length_x*Length_y;

    Phi_bending = 0.0;
    #pragma omp parallel for reduction(+:Phi_bending)
    for(int i=0; i<vertices; i++) {
        energyNode_i(i); // Contribution due to mean curvature and surface area
        phi_vertex_original[i] = phi_vertex[i];
        Phi_bending += phi_vertex[i];
    }
    Phi += Phi_bending;

    // Evaluate Ising model like energy
    Mass = 0;
    Magnet = 0;
    #pragma omp parallel for reduction(+:Mass,Magnet)
    for(int i=0; i<vertices; i++) {
        if(Ising_Array[i] < 2) {
            Mass += Ising_Array[i];
            Magnet += ising_values[Ising_Array[i]];
        }
    }
    Phi -= h_external*Magnet;   
 
    double Phi_magnet = 0;
    #pragma omp parallel for reduction(+:Phi_magnet)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            Phi_magnet -= J_coupling[Ising_Array[i]][Ising_Array[point_neighbor_list[i][j]]]*ising_values[Ising_Array[i]]*ising_values[Ising_Array[point_neighbor_list[i][j]]];
        }
    }
    Phi += Phi_magnet*0.5; // Dividing by 2 as double counting
    Phi_phi = Phi_magnet*0.5;
}
