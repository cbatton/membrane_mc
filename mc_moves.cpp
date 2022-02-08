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
#include "mc_moves.hpp"
using namespace std;

MCMoves::MCMoves(MembraneMC* sys_) {
    // Constructor
    // Assign system to current system
    sys = sys_;
}

MCMoves::~MCMoves() {
    // Destructor
    // Does nothing
}

void MCMoves::DisplaceStep(int vertex_trial, int thread_id) {
    // Pick random site and translate node
    Saru& local_generator = sys->generators[thread_id];
    sys->radii_tri[vertex_trial][0] += lambda*local_generator.d(-1.0,1.0);
    sys->radii_tri[vertex_trial][1] += lambda*local_generator.d(-1.0,1.0);
    sys->radii_tri[vertex_trial][2] += lambda*sys->lengths[1]*local_generator.d(-1.0,1.0);
    // Apply PBC on particles
    sys->radii_tri[vertex_trial][0] -= round(sys->radii_tri[vertex_trial][0]);
    sys->radii_tri[vertex_trial][1] -= round(sys->radii_tri[vertex_trial][1]);
    double phi_diff = 0; 
    double phi_diff_bending = 0;
    double area_diff = 0.0;
	// Loop through neighbor lists
	// Compare versus looping through all for verification
	// Determine index of current location
	int index_x = int(sys->lengths[0]*(sys->radii_tri[vertex_trial][0]+0.5)/sys->nl->box_x);
	int index_y = int(sys->lengths[1]*(sys->radii_tri[vertex_trial][1]+0.5)/sys->nl->box_y);
	int index_z = int((sys->radii_tri[vertex_trial][2]+sys->lengths[2])/sys->nl->box_z);
    if(index_x == sys->nl->nl_x) {
        index_x -= 1;
    }
    if(index_y == sys->nl->nl_y) {
        index_y -= 1;
    }
	int index = index_x + index_y*sys->nl->nl_x + index_z*sys->nl->nl_x*sys->nl->nl_y;
	// Loop through neighboring boxes
	// Check to make sure not counting self case
	for(int i=0; i<sys->nl->neighbors[index].size(); i++) {
		for(int j=0; j<sys->nl->neighbor_list[sys->nl->neighbors[index][i]].size(); j++) {
			// Check particle interactions
            if(vertex_trial != sys->nl->neighbor_list[sys->nl->neighbors[index][i]][j]) {
                double length_neighbor = sys->sim_util->LengthLink(vertex_trial,sys->nl->neighbor_list[sys->nl->neighbors[index][i]][j]);
                if(length_neighbor < 1.0) {
                    phi_diff += pow(10,100);
                }
            }
		}
	}
    // Check to see if particle moved out of bound of checkerboard
    int index_checkerboard_x = floor((sys->lengths[0]*(sys->radii_tri[vertex_trial][0]+0.5)-sys->nl->cell_center_x)/sys->nl->box_x_checkerboard);
	int index_checkerboard_y = floor((sys->lengths[1]*(sys->radii_tri[vertex_trial][1]+0.5)-sys->nl->cell_center_y)/sys->nl->box_y_checkerboard);
    if(index_checkerboard_x == -1) {
        index_checkerboard_x += sys->nl->checkerboard_x;
    }
    if(index_checkerboard_y == -1) {
        index_checkerboard_y += sys->nl->checkerboard_y;
    }
    if(index_checkerboard_x == sys->nl->checkerboard_x) {
        index_checkerboard_x -= 1;
    }
    if(index_checkerboard_y == sys->nl->checkerboard_y) {
        index_checkerboard_y -= 1;
    }
    int index_checkerboard = index_checkerboard_x + index_checkerboard_y*sys->nl->checkerboard_x;
    if(index_checkerboard != sys->nl->checkerboard_index[sys->nl->vertex_trial]) {
        phi_diff += pow(10,100);
    }
    
    // Energy due to bending energy
    // Precompute acceptance/rejectance probabilities
    double chance = local_generator.d();
    double chance_factor = -temp*log(chance);
    if(phi_diff < pow(10,10)) {
        // Energy due to surface area
        for(int i=0; i<sys->point_triangle_list[vertex_trial].size(); i++) {
            int j = sys->point_triangle_list[vertex_trial][i];
            sys->sim_util->AreaNode(j);
            area_total_diff = sys->area_faces[j]-sys->area_faces_original[j];
        }
        // Evaluate curvature energy at node changed and neighboring nodes
        sys->util->EnergyNode(vertex_trial);
        phi_diff_bending += sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->util->EnergyNode(sys->point_neighbor_list[vertex_trial][i]);
            phi_diff_bending += sys->phi_vertex[sys->point_neighbor_list[vertex_trial][i]] - sys->phi_vertex_original[sys->point_neighbor_list[vertex_trial][i]];
        }
        phi_diff += phi_diff_bending;
        // Evaluate energy due to surface tension
        area_diff += sys->sigma_vertex[vertex_trial]-sys->sigma_vertex_original[vertex_trial];
        phi_diff += gamma_surf[sys->ising_array[vertex_trial]]*(sys->sigma_vertex[vertex_trial]-sys->sigma_vertex_original[vertex_trial]);
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            area_diff += sys->sigma_vertex[sys->point_neighbor_list[vertex_trial][i]]-sys->sigma_vertex_original[sys->point_neighbor_list[vertex_trial][i]];
            phi_diff += gamma_surf[sys->ising_array[sys->point_neighbor_list[vertex_trial][i]]]*(sys->sigma_vertex[sys->point_neighbor_list[vertex_trial][i]]-sys->sigma_vertex_original[sys->point_neighbor_list[vertex_trial][i]]);
        }
    }
    // Run probabilities
    if(((chance_factor>phi_diff) && (phi_diff < pow(10,10)))) {
        // Accept move
        // Accept all trial values with for loop
        sys->sim->phi_diff_thread[thread_id][0] += phi_diff;
        sys->sim->phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys->sim->area_diff_thread[thread_id][0] += area_diff;

        sys->radii_tri_original[vertex_trial] = sys->radii_tri[vertex_trial];        

        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->phi_vertex_original[sys->point_neighbor_list[vertex_trial][i]] = sys->phi_vertex[sys->point_neighbor_list[vertex_trial][i]];
        }

        sys->mean_curvature_vertex_original[vertex_trial] = sys->mean_curvature_vertex[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->mean_curvature_vertex_original[sys->point_neighbor_list[vertex_trial][i]] = sys->mean_curvature_vertex[sys->point_neighbor_list[vertex_trial][i]];
        }

        sys->sigma_vertex_original[vertex_trial] = sys->sigma_vertex[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->sigma_vertex_original[sys->point_neighbor_list[vertex_trial][i]] = sys->sigma_vertex[sys->point_neighbor_list[vertex_trial][i]];
        }

        for(int i=0; i<sys->point_triangle_list[vertex_trial].size(); i++) {
            int j = sys->point_triangle_list[vertex_trial][i];
            sys->area_faces_original[j] = sys->area_faces[j];
        }

		// Change neighbor list if new index doesn't match up with old
		if(sys->nl->neighbor_list_index[vertex_trial] != index) {
			// Determine which entry vertex trial was in original index bin and delete
			for(int i=0; i<sys->nl->neighbor_list[sys->nl->neighbor_list_index[vertex_trial]].size(); i++) {
				if(sys->nl->neighbor_list[sys->nl->neighbor_list_index[vertex_trial]][i] == vertex_trial) {
					sys->nl->neighbor_list[sys->nl->neighbor_list_index[vertex_trial]][i] = sys->nl->neighbor_list[sys->nl->neighbor_list_index[vertex_trial]].back();
					sys->nl->neighbor_list[sys->nl->neighbor_list_index[vertex_trial]].pop_back();
					i += sys->nl->neighbor_list[sys->nl->neighbor_list_index[vertex_trial]].size()+10;
				}
			}
			// Add to new bin
			sys->nl->neighbor_list[index].push_back(vertex_trial);
			sys->nl->neighbor_list_index[vertex_trial] = index;
		}
    }
    else {
        steps_rejected_displace_thread[thread_id][0] += 1;

        sys->radii_tri[vertex_trial] = sys->radii_tri_original[vertex_trial];

        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->phi_vertex[sys->point_neighbor_list[vertex_trial][i]] = sys->phi_vertex_original[sys->point_neighbor_list[vertex_trial][i]];
        }

        sys->mean_curvature_vertex[vertex_trial] = sys->mean_curvature_vertex_original[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->mean_curvature_vertex[sys->point_neighbor_list[vertex_trial][i]] = sys->mean_curvature_vertex_original[sys->point_neighbor_list[vertex_trial][i]];
        }

        sys->sigma_vertex[vertex_trial] = sys->sigma_vertex_original[vertex_trial];
        for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
            sys->sigma_vertex[sys->point_neighbor_list[vertex_trial][i]] = sys->sigma_vertex_original[sys->point_neighbor_list[vertex_trial][i]];
        }

        for(int i=0; i<sys->point_triangle_list[vertex_trial].size(); i++) {
            int j = sys->point_triangle_list[vertex_trial][i];
            sys->area_faces[j] = sys->area_faces_original[j];
        }
    }
    steps_tested_displace_thread[thread_id][0] += 1;
}

void MCMoves::TetherCut(int vertex_trial, int thread_id) {
    // Choose link at random, destroy it, and create new link joining other ends of triangle
    // Have to update entries of sys->triangle_list, sys->point_neighbor_list, sys->point_neighbor_triangle, sys->point_triangle_list for this
    // Select random vertex
    Saru& local_generator = sys->generators[thread_id];
	// Reject move if acting on rollers
    // int vertex_trial = 1000;
    // Select random link from avaliable
    int link_trial = local_generator.rand_select(sys->point_neighbor_list[vertex_trial].size()-1);
    int vertex_trial_opposite = sys->point_neighbor_list[vertex_trial][link_trial];

    int triangle_trial[2]; 
    int point_trial[2]; 
    int point_trial_position[2];

    // Find the triangles to be changed using sys->point_neighbor_list
    triangle_trial[0] = sys->point_neighbor_triangle[vertex_trial][link_trial][0];
    triangle_trial[1] = sys->point_neighbor_triangle[vertex_trial][link_trial][1];

    // Find the two other points in the triangles using faces
    for(int i=0; i<2; i++) {
        if ((vertex_trial == sys->triangle_list[triangle_trial[i]][0]) || (vertex_trial_opposite == sys->triangle_list[triangle_trial[i]][0])) {
            if ((vertex_trial == sys->triangle_list[triangle_trial[i]][1]) || (vertex_trial_opposite == sys->triangle_list[triangle_trial[i]][1])) {
                point_trial[i] = sys->triangle_list[triangle_trial[i]][2];
                point_trial_position[i] = 2;
            }
            else {
                point_trial[i] = sys->triangle_list[triangle_trial[i]][1];
                point_trial_position[i] = 1;
            }
        }
        else {
            point_trial[i] = sys->triangle_list[triangle_trial[i]][0];
            point_trial_position[i] = 0;
        }
    }
	int triangle_break[2];
    if((vertex_trial < 0) || (vertex_trial_opposite < 0) || (point_trial[0] < 0) || (point_trial[1] < 0) || (vertex_trial_opposite > sys->vertices) || (point_trial[0] > sys->vertices) || (point_trial[1] > sys->vertices)) {
		steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if point_trial[0] and point_trial[1] are already linked
    for(int i=0; i<point_neighbor_max[point_trial[0]]; i++) {
        if(sys->point_neighbor_list[point_trial[0]][i] == point_trial[1]) {
            steps_rejected_tether_thread[thread_id][0] += 1;
            steps_tested_tether_thread[thread_id][0] += 1;
            return;
        }
    }
    
	// Check to see if the limits for maximum or minimum number of sys->nl->neighbors is exceeded
	if(((sys->point_neighbor_list[vertex_trial].size()-1) == sys->neighbor_min) || ((sys->point_neighbor_list[vertex_trial_opposite].size()-1) == sys->neighbor_min) || ((sys->point_neighbor_list[point_trial[0]].size()+1) == sys->neighbor_max) || ((sys->point_neighbor_list[point_trial[1]].size()+1) == sys->neighbor_max)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
		return;
	}
    
    // Check to see if any of the points are outside of the checkerboard set that vertex_trial is in
    if((sys->nl->checkerboard_index[vertex_trial] != sys->nl->checkerboard_index[vertex_trial_opposite]) || (sys->nl->checkerboard_index[vertex_trial] != sys->nl->checkerboard_index[point_trial[0]]) || (sys->nl->checkerboard_index[vertex_trial] != sys->nl->checkerboard_index[point_trial[1]])) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
		return;
    }

    // Check to see if trial points are too far apart
    double distance_point_trial = sys->sim_util->LengthLink(point_trial[0], point_trial[1]); 
    if ((distance_point_trial > 1.673) || (distance_point_trial < 1.00)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        return;    
    }

    // Calculate detailed balance factor before everything changes
    // Basically, we now acc(v -> w) = gen(w->v) P(w)/gen(v->w) P(v)
    // Probability of gen(v->w) is 1/N*(1/sys->nl->neighbors at vertex trial + 1/sys->nl->neighbors at vertex_trial_opposite)
    // Similar for gen(w->v) except with point trial
    double db_factor = (1.0/(double(sys->point_neighbor_list[point_trial[0]].size())+1.0)+1.0/(double(sys->point_neighbor_list[point_trial[1]].size())+1.0))/(1.0/double(sys->point_neighbor_list[vertex_trial].size())+1.0/double(sys->point_neighbor_list[vertex_trial_opposite].size()));

    // Have all points needed, now just time to change sys->triangle_list, sys->point_neighbor_list, sys->point_neighbor_triangle, sys->point_triangle_list
    // Change triangle_list
    // Check orientation to see if consistent with before
    for(int i=0; i<3; i++) {
        if(sys->triangle_list[triangle_trial[0]][i] == vertex_trial_opposite) {
            sys->triangle_list[triangle_trial[0]][i] = point_trial[1];
            break;
        }
    }
    for(int i=0; i<3; i++) {
        if(sys->triangle_list[triangle_trial[1]][i] == vertex_trial) {
            sys->triangle_list[triangle_trial[1]][i] = point_trial[0];
            break;
        }
    }

    // Change sys->point_neighbor_list and sys->point_neighbor_triangle
    // Delete points
    int placeholder_nl = 0;
    int placeholder_neighbor1 = 0;
    int placeholder_neighbor2 = 0;
    while(placeholder_nl < sys->point_neighbor_list[vertex_trial].size()) {
        if(sys->point_neighbor_list[vertex_trial][placeholder_nl] == vertex_trial_opposite) {
            sys->point_neighbor_list[vertex_trial][placeholder_nl] = sys->point_neighbor_list[vertex_trial].back()'
            sys->point_neighbor_triangle[vertex_trial][placeholder_nl] = sys->point_neighbor_triangle[vertex_trial].back();
            placeholder_neighbor1 = placeholder_nl;
            sys->point_neighbor_list[vertex_trial].pop_back();
            sys->point_neighbor_triangle[vertex_trial].pop_back();
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    placeholder_nl = 0;
    while(placeholder_nl < point_neighbor_max[vertex_trial_opposite]) {
        if(sys->point_neighbor_list[vertex_trial_opposite][placeholder_nl] == vertex_trial) {
            sys->point_neighbor_list[vertex_trial_opposite][placeholder_nl] = sys->point_neighbor_list[vertex_trial_opposite].back();
            sys->point_neighbor_triangle[vertex_trial_opposite][placeholder_nl] = sys->point_neighbor_triangle[vertex_trial_opposite].back();
            placeholder_neighbor2 = placeholder_nl;
            sys->point_neighbor_list[vertex_trial_opposite].pop_back();
            sys->point_neighbor_triangle[vertex_trial_opposite].pop_back();
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    // Add points
    vector<int> points_add{triangle_trial[0],triangle_trial[1]};
    sys->point_neighbor_list[point_trial[0]].push_back(point_trial[1]);
    sys->point_neighbor_list[point_trial[1]].push_back(point_trial[0]);
    sys->point_neighbor_triangle[point_trial[0]].push_back(points_add);
    sys->point_neighbor_triangle[point_trial[1]].push_back(points_add);

    // Note that the definition of triangle_trial[0] and triangle_trial[1] have changed
    // Need to modify sys->point_neighbor_triangle entries between vertex_trial and point_trial[1]
    // and vertex_trial_opposite and point_trial[0] to swap triangle_trial[1] to triangle_trial[0]
    // and triangle_trial[0] to triangle_trial[1] respectively
    // Placeholder values so places needed are saved
    int placeholder_remake[4] = {0,0,0,0};
    int placeholder_remake_01[4] = {0,0,0,0};
    // vertex_trial and point_trial[1]
    for(int i=0; i<sys->point_neighbor_list[vertex_trial].size(); i++) {
        if(sys->point_neighbor_list[vertex_trial][i] == point_trial[1]) {
            placeholder_remake[0] = i;
            if(sys->point_neighbor_triangle[vertex_trial][i][0] == triangle_trial[1]) {
                sys->point_neighbor_triangle[vertex_trial][i][0] = triangle_trial[0];
                placeholder_remake_01[0] = 0;
            }
            else if(sys->point_neighbor_triangle[vertex_trial][i][1] == triangle_trial[1]) {
                sys->point_neighbor_triangle[vertex_trial][i][1] = triangle_trial[0];
                placeholder_remake_01[0] = 1;
            }
        }
    }
    for(int i=0; i<point_neighbor_max[point_trial[1]]; i++) {
        if(sys->point_neighbor_list[point_trial[1]][i] == vertex_trial) {
            placeholder_remake[1] = i;
            if(sys->point_neighbor_triangle[point_trial[1]][i][0] == triangle_trial[1]) {
                sys->point_neighbor_triangle[point_trial[1]][i][0] = triangle_trial[0];
                placeholder_remake_01[1] = 0;
            }
            else if(sys->point_neighbor_triangle[point_trial[1]][i][1] == triangle_trial[1]) {
                sys->point_neighbor_triangle[point_trial[1]][i][1] = triangle_trial[0];
                placeholder_remake_01[1] = 1;
            }
        }
    }
    // vertex_trial_opposite and point_trial[0]
    for(int i=0; i<point_neighbor_max[vertex_trial_opposite]; i++) {
        if(sys->point_neighbor_list[vertex_trial_opposite][i] == point_trial[0]) {
            placeholder_remake[2] = i;
            if(sys->point_neighbor_triangle[vertex_trial_opposite][i][0] == triangle_trial[0]) {
                sys->point_neighbor_triangle[vertex_trial_opposite][i][0] = triangle_trial[1];
                placeholder_remake_01[2] = 0;
            }
            else if(sys->point_neighbor_triangle[vertex_trial_opposite][i][1] == triangle_trial[0]) {
                sys->point_neighbor_triangle[vertex_trial_opposite][i][1] = triangle_trial[1];
                placeholder_remake_01[2] = 1;
            }
        }
    }
    for(int i=0; i<point_neighbor_max[point_trial[0]]; i++) {
        if(sys->point_neighbor_list[point_trial[0]][i] == vertex_trial_opposite) {
            placeholder_remake[3] = i;
            if(sys->point_neighbor_triangle[point_trial[0]][i][0] == triangle_trial[0]) {
                sys->point_neighbor_triangle[point_trial[0]][i][0] = triangle_trial[1];
                placeholder_remake_01[3] = 0;
            }
            else if(sys->point_neighbor_triangle[point_trial[0]][i][1] == triangle_trial[0]) {
                sys->point_neighbor_triangle[point_trial[0]][i][1] = triangle_trial[1];
                placeholder_remake_01[3] = 1;
            }
        }
    }

    // Change sys->point_triangle_list
    placeholder_nl = 0;
    int placeholder_triangle1 = 0;
    int placeholder_triangle2 = 0;
    while(placeholder_nl < sys->point_triangle_list[vertex_trial].size()) {
        if(sys->point_triangle_list[vertex_trial][placeholder_nl] == triangle_trial[1]) {
            sys->point_triangle_list[vertex_trial][placeholder_nl] = sys->point_triangle_list[vertex_trial].back();
            placeholder_triangle1 = placeholder_nl;
            sys->point_triangle_list[vertex_trial].pop_back();
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    placeholder_nl = 0;
    while(placeholder_nl < sys->point_triangle_list[vertex_trial_opposite].size()) {
        if(sys->point_triangle_list[vertex_trial_opposite][placeholder_nl] == triangle_trial[0]) {
            sys->point_triangle_list[vertex_trial_opposite][placeholder_nl] = sys->point_triangle_list[vertex_trial_opposite].back()
            placeholder_triangle2 = placeholder_nl;
            sys->point_triangle_list[vertex_trial_opposite].pop_back();
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    // Add points
    sys->point_triangle_list[point_trial[0]].push_back(triangle_trial[1]);
    sys->point_triangle_list[point_trial[1]].push_back(triangle_trial[0]);

    // Evaluate energy difference
    // Evaluated at four nodes from two triangles changed
    double phi_diff = 0; 
    double phi_diff_bending = 0;
    double phi_diff_phi = 0;
    // Energy due to mean curvature
    sys->util->EnergyNode(vertex_trial);
    sys->util->EnergyNode(vertex_trial_opposite);
    sys->util->EnergyNode(point_trial[0]);
    sys->util->EnergyNode(point_trial[1]);

    phi_diff_bending += sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    phi_diff += gamma_surf[sys->ising_array[vertex_trial]]*(sys->sigma_vertex[vertex_trial]-sys->sigma_vertex_original[vertex_trial]);

    phi_diff_bending += sys->phi_vertex[vertex_trial_opposite] - sys->phi_vertex_original[vertex_trial_opposite];
    phi_diff += gamma_surf[sys->ising_array[vertex_trial_opposite]]*(sys->sigma_vertex[vertex_trial_opposite]-sys->sigma_vertex_original[vertex_trial_opposite]);

    phi_diff_bending += sys->phi_vertex[point_trial[0]] - sys->phi_vertex_original[point_trial[0]];
    phi_diff += gamma_surf[sys->ising_array[point_trial[0]]]*(sys->sigma_vertex[point_trial[0]]-sys->sigma_vertex_original[point_trial[0]]);

    phi_diff_bending += sys->phi_vertex[point_trial[1]] - sys->phi_vertex_original[point_trial[1]];
    phi_diff += gamma_surf[sys->ising_array[point_trial[1]]]*(sys->sigma_vertex[point_trial[1]]-sys->sigma_vertex_original[point_trial[1]]);

    phi_diff += phi_diff_bending;

    // Area change of triangles
    sys->sim_util->AreaNode(triangle_trial[0]);
    sys->sim_util->AreaNode(triangle_trial[1]);

    // Energy due to different Ising interactions
	phi_diff_phi += sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[vertex_trial_opposite]]*sys->ising_values[sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array[vertex_trial_opposite]];
	phi_diff_phi -= sys->j_coupling[sys->ising_array[point_trial[0]]][sys->ising_array[point_trial[1]]]*sys->ising_values[sys->ising_array[point_trial[0]]]*sys->ising_values[sys->ising_array[point_trial[1]]];

    phi_diff += phi_diff_phi;
    double chance = local_generator.d();
    double chance_factor = -temp*log(chance/db_factor);
    bool accept = false;

    if((accept == true) || ((chance_factor>phi_diff) && (phi_diff < pow(10,10)))) {
        // Accept move
        // New way that uses energy loop
        // Accept all trial values with for loop
        sys->sim->phi_diff_thread[thread_id][0] += phi_diff;
        sys->sim->phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys->sim->phi_phi_diff_thread[thread_id][0] += phi_diff_phi;

        // Update original values
        // Have all points needed, now just time to change triangle_list, sys->point_neighbor_list, link_triangle_list, sys->point_triangle_list 
        // Change triangle_list
        triangle_list_original[triangle_trial[0]] = triangle_list[triangle_trial[0]];
        triangle_list_original[triangle_trial[1]] = triangle_list[triangle_trial[1]];

        // Change sys->point_neighbor_list
        // Delete points
        sys->point_neighbor_list_original[vertex_trial] = sys->point_neighbor_list[vertex_trial];
        sys->point_neighbor_triangle_original[vertex_trial] = sys->point_neighbor_triangle[vertex_trial];
        sys->point_neighbor_list_original[vertex_trial_opposite] = sys->point_neighbor_list[vertex_trial_opposite];
        sys->point_neighbor_triangle_original[vertex_trial_opposite] = sys->point_neighbor_triangle[vertex_trial_opposite];

        // Add points
        sys->point_neighbor_list_original[point_trial[0]] = sys->point_neighbor_list[point_trial[0]];
        sys->point_neighbor_triangle_original[point_trial[0]] = sys->point_neighbor_triangle[point_trial[0]];
        sys->point_neighbor_list_original[point_trial[1]] = sys->point_neighbor_list[point_trial[1]];
        sys->point_neighbor_triangle_original[point_trial[1]] = sys->point_neighbor_triangle[point_trial[1]];

        // Change sys->point_triangle_list
        sys->point_triangle_list_original[vertex_trial] = sys->point_triangle_list[vertex_trial];
        sys->point_triangle_list_original[vertex_trial_opposite] = sys->point_triangle_list[vertex_trial_opposite];
        sys->point_triangle_list_original[point_trial[0]] = sys->point_triangle_list[point_trial[0]];
        sys->point_triangle_list_original[point_trial[1]] = sys->point_triangle_list[point_trial[1]];

        // Update energy values
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
        sys->phi_vertex_original[vertex_trial_opposite] = sys->phi_vertex[vertex_trial_opposite];
        sys->phi_vertex_original[point_trial[0]] = sys->phi_vertex[point_trial[0]];
        sys->phi_vertex_original[point_trial[1]] = sys->phi_vertex[point_trial[1]];
        // Update mean curvature
        sys->mean_curvature_vertex_original[vertex_trial] = sys->mean_curvature_vertex[vertex_trial];
        sys->mean_curvature_vertex_original[vertex_trial_opposite] = sys->mean_curvature_vertex[vertex_trial_opposite];
        sys->mean_curvature_vertex_original[point_trial[0]] = sys->mean_curvature_vertex[point_trial[0]];
        sys->mean_curvature_vertex_original[point_trial[1]] = sys->mean_curvature_vertex[point_trial[1]];
        // Update sigma values
        sys->sigma_vertex_original[vertex_trial] = sys->sigma_vertex[vertex_trial];
        sys->sigma_vertex_original[vertex_trial_opposite] = sys->sigma_vertex[vertex_trial_opposite];
        sys->sigma_vertex_original[point_trial[0]] = sys->sigma_vertex[point_trial[0]];
        sys->sigma_vertex_original[point_trial[1]] = sys->sigma_vertex[point_trial[1]];

        // Update area of phases
        double area_total_diff = 0.0;
        area_total_diff += sys->area_faces[triangle_trial[0]] - sys->area_faces_original[triangle_trial[0]];
        area_total_diff += sys->area_faces[triangle_trial[1]] - sys->area_faces_original[triangle_trial[1]];
        sys->sim->area_diff_thread[thread_id][0] += area_total_diff;

        sys->area_faces_original[triangle_trial[0]] = sys->area_faces[triangle_trial[0]];
        sys->area_faces_original[triangle_trial[1]] = sys->area_faces[triangle_trial[1]]; 
            
    }
    else {
        // cout << "Reject move at " << vertex_trial << endl;
        steps_rejected_tether_thread[thread_id][0] += 1;
        // Change new values to original values
        // Have all points needed, now just time to change triangle_list, sys->point_neighbor_list, link_triangle_list, sys->point_triangle_list
        // Change triangle_list
        triangle_list[triangle_trial[0]] = triangle_list_original[triangle_trial[0]];
        triangle_list[triangle_trial[1]] = triangle_list_original[triangle_trial[1]];

        // Change sys->point_neighbor_list
        // Delete points
        sys->point_neighbor_list[vertex_trial] = sys->point_neighbor_list_original[vertex_trial];
        sys->point_neighbor_triangle[vertex_trial] = sys->point_neighbor_triangle_original[vertex_trial];
        sys->point_neighbor_list[vertex_trial_opposite] = sys->point_neighbor_list_original[vertex_trial_opposite];
        sys->point_neighbor_triangle[vertex_trial_opposite] = sys->point_neighbor_triangle_original[vertex_trial_opposite];

        // Add points
        sys->point_neighbor_list[point_trial[0]] = sys->point_neighbor_list_original[point_trial[0]];
        sys->point_neighbor_triangle[point_trial[0]] = sys->point_neighbor_triangle_original[point_trial[0]];
        sys->point_neighbor_list[point_trial[1]] = sys->point_neighbor_list_original[point_trial[1]];
        sys->point_neighbor_triangle[point_trial[1]] = sys->point_neighbor_triangle_original[point_trial[1]];

        // Change sys->point_triangle_list
        sys->point_triangle_list[vertex_trial] = sys->point_triangle_list_original[vertex_trial];
        sys->point_triangle_list[vertex_trial_opposite] = sys->point_triangle_list_original[vertex_trial_opposite];
        sys->point_triangle_list[point_trial[0]] = sys->point_triangle_list_original[point_trial[0]];
        sys->point_triangle_list[point_trial[1]] = sys->point_triangle_list_original[point_trial[1]];

        // Update energy values
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];
        sys->phi_vertex[vertex_trial_opposite] = sys->phi_vertex_original[vertex_trial_opposite];
        sys->phi_vertex[point_trial[0]] = sys->phi_vertex_original[point_trial[0]];
        sys->phi_vertex[point_trial[1]] = sys->phi_vertex_original[point_trial[1]];
        // Update mean curvature
        sys->mean_curvature_vertex[vertex_trial] = sys->mean_curvature_vertex_original[vertex_trial];
        sys->mean_curvature_vertex[vertex_trial_opposite] = sys->mean_curvature_vertex_original[vertex_trial_opposite];
        sys->mean_curvature_vertex[point_trial[0]] = sys->mean_curvature_vertex_original[point_trial[0]];
        sys->mean_curvature_vertex[point_trial[1]] = sys->mean_curvature_vertex_original[point_trial[1]];
        // Update sigma values
        sys->sigma_vertex[vertex_trial] = sys->sigma_vertex_original[vertex_trial];
        sys->sigma_vertex[vertex_trial_opposite] = sys->sigma_vertex_original[vertex_trial_opposite];
        sys->sigma_vertex[point_trial[0]] = sys->sigma_vertex_original[point_trial[0]];
        sys->sigma_vertex[point_trial[1]] = sys->sigma_vertex_original[point_trial[1]];
   
        // Update area of phases
        sys->area_faces[triangle_trial[0]] = sys->area_faces_original[triangle_trial[0]];
        sys->area_faces[triangle_trial[1]] = sys->area_faces_original[triangle_trial[1]]; 

        
    }
    steps_tested_tether_thread[thread_id][0] += 1;
}

void MCMoves::ChangeMassNonCon(int vertex_trial, int thread_id) {
    // Pick random site and change the Ising array value
    // Non-mass conserving
    Saru& local_generator = generators[thread_id];
    // Have to have implementation that chooses from within the same checkerboard set
    if(sys->ising_array[vertex_trial] == 2) {
        // If protein node is selected, do a moveProtein_gen move instead
        moveProtein_gen(vertex_trial, thread_id);
        return;
	} 
    // Change spin
    int sys->ising_array_trial = 0;
    if(sys->ising_array[vertex_trial] == 0){
        sys->ising_array_trial = 1;
    }

    double phi_diff_phi = 0;

    for(int j=0; j<sys->point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array[vertex_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array_trial];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    double phi_diff_mag = -h_external*(sys->ising_values[sys->ising_array_trial]-sys->ising_values[sys->ising_array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = sys->mean_curvature_vertex[vertex_trial]-spon_curv[sys->ising_array_trial];
    sys->phi_vertex[vertex_trial] = k_b[sys->ising_array_trial]*sys->sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double phi_diff_bending = sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    double phi_diff = phi_diff_phi+phi_diff_bending+phi_diff_mag;
    double chance = local_generator.d();
    if(chance<exp(-phi_diff/T)){
        sys->sim->phi_diff_thread[thread_id][0] += phi_diff;
        sys->sim->phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        phi_sys->sim->phi_diff_thread[thread_id][0] += phi_diff_phi;
        mass_diff_thread[thread_id][0] += (sys->ising_array_trial-sys->ising_array[vertex_trial]);
        magnet_diff_thread[thread_id][0] += (sys->ising_values[sys->ising_array_trial]-sys->ising_values[sys->ising_array[vertex_trial]]);
        sys->ising_array[vertex_trial] = sys->ising_array_trial;
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;
}

void MCMoves::ChangeMassNonConGL(int vertex_trial) {
    // Pick random site and change the Ising array value per Glauber dynamics
    // Non-mass conserving
    Saru& local_generator = generators[omp_get_thread_num()];
    while (sys->ising_array[vertex_trial] == 2) {
		vertex_trial = local_generator.rand_select(sys->vertices-1);
	}
    // Change spin
    int sys->ising_array_trial = 0;
    if(sys->ising_array[vertex_trial] == 0){
        sys->ising_array_trial = 1;
    }

    double phi_magnet = 0;

    for(int j=0; j<sys->point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array[vertex_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array_trial];
        phi_magnet -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    phi_magnet -= h_external*(sys->ising_values[sys->ising_array_trial]-sys->ising_values[sys->ising_array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = sys->mean_curvature_vertex[vertex_trial]-spon_curv[sys->ising_array_trial];
    sys->phi_vertex[vertex_trial] = k_b[sys->ising_array_trial]*sys->sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double phi_diff = phi_magnet + sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    double chance = local_generator.d();
    if(chance<(1.0/(1.0+exp(phi_diff/T)))) {
        #pragma omp atomic
        phi += phi_diff;
        #pragma omp atomic
        Mass += (sys->ising_array_trial-sys->ising_array[vertex_trial]);
        #pragma omp atomic
        Magnet += (sys->ising_values[sys->ising_array_trial]-sys->ising_values[sys->ising_array[vertex_trial]]);
        sys->ising_array[vertex_trial] = sys->ising_array_trial;
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
    }
    
    else {
        #pragma omp atomic
        steps_rejected_mass += 1;
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];    
    }
    #pragma omp atomic
    steps_tested_mass += 1;
}

void MCMoves::ChangeMassCon(int vertex_trial, int thread_id) {
    // Pick random site and attemp to swap Ising array value with array in random nearest neighbor direction
    // Mass conserving
    Saru& local_generator = generators[thread_id];
    if(sys->ising_array[vertex_trial] == 2) {
        // If protein node is selected, do a moveProtein_gen move instead
        moveProtein_gen(vertex_trial, thread_id);
        return;
	} 
    // Pick random direction
    int link_trial = local_generator.rand_select(sys->point_neighbor_list[vertex_trial].size()-1);
    int vertex_trial_opposite = sys->point_neighbor_list[vertex_trial][link_trial];

    // For now reject if the neighboring sites have the same array value or protein type
    if(sys->ising_array[vertex_trial] == sys->ising_array[vertex_trial_opposite]) {
        steps_rejected_mass_thread[thread_id][0] += 1;
        steps_tested_mass_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if vertex_trial_opposite is not in same checkerboard set
    if(checkerboard_index[vertex_trial] != checkerboard_index[vertex_trial_opposite]) {
        steps_rejected_mass_thread[thread_id][0] += 1;
    	steps_tested_mass_thread[thread_id][0] += 1;
		return;
    }

    // Set trial values
    int sys->ising_array_trial_1 = sys->ising_array[vertex_trial_opposite];
    int sys->ising_array_trial_2 = sys->ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_diff_phi = 0;
    for(int j=0; j<sys->point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array[vertex_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array_trial_1];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys->mean_curvature_vertex[vertex_trial]-spon_curv[sys->ising_array_trial_1];
    sys->phi_vertex[vertex_trial] = k_b[sys->ising_array_trial_1]*sys->sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_opposite]; j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial_opposite]][sys->ising_array[sys->point_neighbor_list[vertex_trial_opposite][j]]]*sys->ising_values[sys->ising_array[vertex_trial_opposite]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[sys->point_neighbor_list[vertex_trial_opposite][j]]]*sys->ising_values[sys->ising_array_trial_2];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial_opposite][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys->mean_curvature_vertex[vertex_trial_opposite]-spon_curv[sys->ising_array_trial_2];
    sys->phi_vertex[vertex_trial_opposite] = k_b[sys->ising_array_trial_2]*sys->sigma_vertex[vertex_trial_opposite]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    phi_diff_phi += (sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[vertex_trial_opposite]]*sys->ising_values[sys->ising_array_trial_1]-sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[vertex_trial_opposite]]*sys->ising_values[sys->ising_array[vertex_trial]])*sys->ising_values[sys->ising_array[vertex_trial_opposite]];
    phi_diff_phi += (sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array_trial_2]-sys->j_coupling[sys->ising_array[vertex_trial_opposite]][sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array[vertex_trial_opposite]])*sys->ising_values[sys->ising_array[vertex_trial]];

    double phi_diff_bending = sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    phi_diff_bending += sys->phi_vertex[vertex_trial_opposite] - sys->phi_vertex_original[vertex_trial_opposite];
    double phi_diff = phi_diff_bending+phi_diff_phi;
    double chance = local_generator.d();
    if(chance<exp(-phi_diff/T)){
        sys->sim->phi_diff_thread[thread_id][0] += phi_diff;
        sys->sim->phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        phi_sys->sim->phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys->ising_array[vertex_trial] = sys->ising_array_trial_1;
        sys->ising_array[vertex_trial_opposite] = sys->ising_array_trial_2;
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
        sys->phi_vertex_original[vertex_trial_opposite] = sys->phi_vertex[vertex_trial_opposite];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];
        sys->phi_vertex[vertex_trial_opposite] = sys->phi_vertex_original[vertex_trial_opposite];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;

}

void MCMoves::ChangeMassConNL() {
    // Pick random site and attemp to swap Ising array value with another nonlocal site
    // Mass conserving
    // Can't think of a good parallel implementation for now
    Saru& local_generator = generators[omp_get_thread_num()];
    int vertex_trial = local_generator.rand_select(sys->vertices-1);
    // Pick random site with opposite spin
    int vertex_trial_2 = local_generator.rand_select(sys->vertices-1);

    // Keep generating new trial values if Ising values are the same
    while((sys->ising_array[vertex_trial] == sys->ising_array[vertex_trial_2]) || (sys->ising_array[vertex_trial] == 2) || (sys->ising_array[vertex_trial_2] == 2)) {
    	vertex_trial = local_generator.rand_select(sys->vertices-1);
        vertex_trial_2 = local_generator.rand_select(sys->vertices-1);   		 
    }    

    // Set trial values
    int sys->ising_array_trial_1 = sys->ising_array[vertex_trial_2];
    int sys->ising_array_trial_2 = sys->ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_magnet = 0;
    for(int j=0; j<sys->point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array[vertex_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array_trial_1];
        phi_magnet -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys->mean_curvature_vertex[vertex_trial]-spon_curv[sys->ising_array_trial_1];
    sys->phi_vertex[vertex_trial] = k_b[sys->ising_array_trial_1]*sys->sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_2]; j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial_2]][sys->ising_array[sys->point_neighbor_list[vertex_trial_2][j]]]*sys->ising_values[sys->ising_array[vertex_trial_2]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[sys->point_neighbor_list[vertex_trial_2][j]]]*sys->ising_values[sys->ising_array_trial_2];
        phi_magnet -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys->mean_curvature_vertex[vertex_trial_2]-spon_curv[sys->ising_array_trial_2];
    sys->phi_vertex[vertex_trial_2] = k_b[sys->ising_array_trial_2]*sys->sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    // Now check for self contribution
    // First check to see if sites are neighboring
    int check_double_count = link_triangle_test(vertex_trial, vertex_trial_2);
    if(check_double_count == 1) {
        phi_magnet += (sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[vertex_trial_2]]*sys->ising_values[sys->ising_array_trial_1]-sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[vertex_trial_2]]*sys->ising_values[sys->ising_array[vertex_trial]])*sys->ising_values[sys->ising_array[vertex_trial_2]];
        phi_magnet += (sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array_trial_2]-sys->j_coupling[sys->ising_array[vertex_trial_2]][sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array[vertex_trial_2]])*sys->ising_values[sys->ising_array[vertex_trial]];
    }    

    double phi_diff = phi_magnet + sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    phi_diff += sys->phi_vertex[vertex_trial_2] - sys->phi_vertex_original[vertex_trial_2];
    double chance = local_generator.d();
    if(chance<exp(-phi_diff/T)){
        phi += phi_diff;
        sys->ising_array[vertex_trial] = sys->ising_array_trial_1;
        sys->ising_array[vertex_trial_2] = sys->ising_array_trial_2;
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
        sys->phi_vertex_original[vertex_trial_2] = sys->phi_vertex[vertex_trial_2];
    }
    
    else {
        steps_rejected_mass += 1;
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];
        sys->phi_vertex[vertex_trial_2] = sys->phi_vertex_original[vertex_trial_2];    
    }
    steps_tested_mass += 1;

}

void MCMoves::MoveProteinGen(int vertex_trial, int thread_id) {
    // Pick random protein and attempt to move it in the y-direction
    // As protein's not merging, don't let them
    Saru& local_generator = generators[thread_id];
    // cout << "Protein trial " << protein_trial << " ";
    // Pick direction
    // Instead just go with one it's sys->nl->neighbors
    int direction_trial = local_generator.rand_select(sys->point_neighbor_list[vertex_trial].size()-1);
    int center_trial = sys->point_neighbor_list[vertex_trial][direction_trial]; 
    // Reject if about to swap with another protein of the same type
    if((protein_node[vertex_trial] != -1) && (protein_node[vertex_trial] == protein_node[center_trial])) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Reject if not in the same checkerboard set
    if(checkerboard_index[vertex_trial] != checkerboard_index[center_trial]) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Have to break down into cases
    int case_vertex = protein_node[vertex_trial];
    int case_center = protein_node[center_trial];

    // Energetics of swapping those two
    // Set trial values
    int sys->ising_array_trial_1 = sys->ising_array[center_trial];
    int sys->ising_array_trial_2 = sys->ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_diff_phi = 0;
    for(int j=0; j<sys->point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array[vertex_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array_trial_1];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys->mean_curvature_vertex[vertex_trial]-spon_curv[sys->ising_array_trial_1];
    sys->phi_vertex[vertex_trial] = k_b[sys->ising_array_trial_1]*sys->sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[center_trial]; j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[center_trial]][sys->ising_array[sys->point_neighbor_list[center_trial][j]]]*sys->ising_values[sys->ising_array[center_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[sys->point_neighbor_list[center_trial][j]]]*sys->ising_values[sys->ising_array_trial_2];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[center_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys->mean_curvature_vertex[center_trial]-spon_curv[sys->ising_array_trial_2];
    sys->phi_vertex[center_trial] = k_b[sys->ising_array_trial_2]*sys->sigma_vertex[center_trial]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    phi_diff_phi += (sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[center_trial]]*sys->ising_values[sys->ising_array_trial_1]-sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[center_trial]]*sys->ising_values[sys->ising_array[vertex_trial]])*sys->ising_values[sys->ising_array[center_trial]];
    phi_diff_phi += (sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array_trial_2]-sys->j_coupling[sys->ising_array[center_trial]][sys->ising_array[vertex_trial]]*sys->ising_values[sys->ising_array[center_trial]])*sys->ising_values[sys->ising_array[vertex_trial]];
    
    double phi_diff_bending = sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    phi_diff_bending += sys->phi_vertex[center_trial] - sys->phi_vertex_original[center_trial];
    double chance = local_generator.d();
    double db_factor = double(sys->point_neighbor_list[vertex_trial].size())/double(point_neighbor_max[center_trial]);
    double chance_factor = -T*log(chance/db_factor);
    double phi_diff = phi_diff_phi+phi_diff_bending;

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(chance_factor>phi_diff) {
        accept = true;
    }

    // cout << phi_magnet << endl;
    // cout << chance << " " << exp(-phi_magnet/T) << endl;
    if(accept == true) {
        sys->sim->phi_diff_thread[thread_id][0] += phi_diff;
        sys->sim->phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        phi_sys->sim->phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys->ising_array[vertex_trial] = sys->ising_array_trial_1;
        sys->ising_array[center_trial] = sys->ising_array_trial_2;
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
        sys->phi_vertex_original[center_trial] = sys->phi_vertex[center_trial];
        protein_node[vertex_trial] = case_center;
        protein_node[center_trial] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];
        sys->phi_vertex[center_trial] = sys->phi_vertex_original[center_trial];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void MCMoves::MoveProteinNL(int vertex_trial, int vertex_trial_2, int thread_id) {
    // Pick two random nodes, see if one is a protein and attempt to swap if so
    Saru& local_generator = generators[thread_id];
    // cout << "Protein trial " << protein_trial << " ";
    // Reject if about to swap with another protein of the same type
    if(((protein_node[vertex_trial] == -1) || (protein_node[vertex_trial_2] == -1)) && (protein_node[vertex_trial] == protein_node[vertex_trial_2])) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Have to break down into cases
    int case_vertex = protein_node[vertex_trial];
    int case_center = protein_node[vertex_trial_2];

    // Energetics of swapping those two
    // Set trial values
    int sys->ising_array_trial_1 = sys->ising_array[vertex_trial_2];
    int sys->ising_array_trial_2 = sys->ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_diff_phi = 0;
    for(int j=0; j<sys->point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial]][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array[vertex_trial]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_1][sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]]*sys->ising_values[sys->ising_array_trial_1];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys->mean_curvature_vertex[vertex_trial]-spon_curv[sys->ising_array_trial_1];
    sys->phi_vertex[vertex_trial] = k_b[sys->ising_array_trial_1]*sys->sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_2]; j++) {
        double Site_diff = sys->j_coupling[sys->ising_array[vertex_trial_2]][sys->ising_array[sys->point_neighbor_list[vertex_trial_2][j]]]*sys->ising_values[sys->ising_array[vertex_trial_2]];
        double Site_diff_2 = sys->j_coupling[sys->ising_array_trial_2][sys->ising_array[sys->point_neighbor_list[vertex_trial_2][j]]]*sys->ising_values[sys->ising_array_trial_2];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys->ising_values[sys->ising_array[sys->point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys->mean_curvature_vertex[vertex_trial_2]-spon_curv[sys->ising_array_trial_2];
    sys->phi_vertex[vertex_trial_2] = k_b[sys->ising_array_trial_2]*sys->sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    double phi_diff_bending = sys->phi_vertex[vertex_trial] - sys->phi_vertex_original[vertex_trial];
    phi_diff_bending += sys->phi_vertex[vertex_trial_2] - sys->phi_vertex_original[vertex_trial_2];
    double chance = local_generator.d();
    double chance_factor = -T*log(chance);
    double phi_diff = phi_diff_phi+phi_diff_bending;

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(chance_factor>phi_diff) {
        accept = true;
    }

    // cout << phi_magnet << endl;
    // cout << chance << " " << exp(-phi_magnet/T) << endl;
    if(accept == true) {
        sys->sim->phi_diff_thread[thread_id][0] += phi_diff;
        sys->sim->phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        phi_sys->sim->phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys->ising_array[vertex_trial] = sys->ising_array_trial_1;
        sys->ising_array[vertex_trial_2] = sys->ising_array_trial_2;
        sys->phi_vertex_original[vertex_trial] = sys->phi_vertex[vertex_trial];
        sys->phi_vertex_original[vertex_trial_2] = sys->phi_vertex[vertex_trial_2];
        protein_node[vertex_trial] = case_center;
        protein_node[vertex_trial_2] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        sys->phi_vertex[vertex_trial] = sys->phi_vertex_original[vertex_trial];
        sys->phi_vertex[vertex_trial_2] = sys->phi_vertex_original[vertex_trial_2];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void MCMoves::ChangeArea() {
// Attempt to change sys->lengths[0] and sys->lengths[1] uniformly
// Going with logarthmic changes in area for now
// Naw, let's try discrete steps
// But discrete changes are also legitimate
    // Select trial scale_xy factor
    // double log_scale_xy_trial = log(scale_xy)+lambda_scale*randNeg1to1(generator);
    // double scale_xy_trial = exp(log_scale_xy_trial);
    chrono::steady_clock::time_point t1_area;
    chrono::steady_clock::time_point t2_area;
    t1_area = chrono::steady_clock::now();

    double scale_xy_trial = scale_xy+lambda_scale*generator.d(-1.0,1.0);
    if(scale_xy_trial <= 0.0) {
        steps_rejected_area++;
        steps_tested_area++;
        return;
    }
    sys->lengths[0] = sys->lengths[0]_base*scale_xy_trial;
    sys->lengths[1] = sys->lengths[1]_base*scale_xy_trial;
    t2_area = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_area-t1_area;
    time_storage_area[0] += time_span.count();
    // Reform neighbor list
    t1_area = chrono::steady_clock::now();
    generateNeighborList();
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[1] += time_span.count();
    // Store original values
    t1_area = chrono::steady_clock::now();
    double phi_ = phi;
    double phi_bending_ = phi_bending;
    double phi_phi_ = phi_phi;
    double area_total_ = area_total;
    // Recompute energy
    // Note that this version doesn't override all variables
    initializeEnergy_scale();
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[2] += time_span.count();
    // Now accept/reject
    t1_area = chrono::steady_clock::now();
    double chance = generator.d();
    double phi_diff = phi-phi_-T*2*sys->vertices*log(scale_xy_trial/scale_xy);
    if((chance<exp(-phi_diff/T)) && (phi_diff < pow(10,10))) {
        scale_xy = scale_xy_trial;
        sys->lengths[0]_old = sys->lengths[0];
        sys->lengths[1]_old = sys->lengths[1];
        sys->nl->box_x = sys->lengths[0]/double(sys->nl->nl_x); 
        sys->nl->box_y = sys->lengths[1]/double(sys->nl->nl_y); 
        #pragma omp parallel for
        for(int i=0; i<sys->vertices; i++) {
            sys->phi_vertex_original[i] = sys->phi_vertex[i];
        }
        #pragma omp parallel for
        for(int i=0; i<faces; i++) {
            sys->area_faces_original[i] = sys->area_faces[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys->vertices; i++) {
            sys->mean_curvature_vertex_original[i] = sys->mean_curvature_vertex[i]; 
        }
        #pragma omp parallel for
        for(int i=0; i<sys->vertices; i++) {
            sys->sigma_vertex_original[i] = sys->sigma_vertex[i]; 
        }
    }
    else {
        steps_rejected_area++;
        phi = phi_;
        phi_bending = phi_bending_;
        phi_phi = phi_phi_;
        area_total = area_total_;
        sys->lengths[0] = sys->lengths[0]_old;
        sys->lengths[1] = sys->lengths[1]_old;
        generateNeighborList();
        sys->nl->box_x = sys->lengths[0]/double(sys->nl->nl_x); 
        sys->nl->box_y = sys->lengths[1]/double(sys->nl->nl_y); 
        #pragma omp parallel for
        for(int i=0; i<sys->vertices; i++) {
            sys->phi_vertex[i] = sys->phi_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<faces; i++) {
            sys->area_faces[i] = sys->area_faces_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys->vertices; i++) {
            sys->mean_curvature_vertex[i] = sys->mean_curvature_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys->vertices; i++) {
            sys->sigma_vertex[i] = sys->sigma_vertex_original[i];
        }
    }
    steps_tested_area++;
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[3] += time_span.count();
}
