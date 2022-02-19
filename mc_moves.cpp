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

MCMoves::MCMoves(double lambda, double lambda_scale, int nl_move_start, int max_threads) : lambda(lambda), lambda_scale(lambda_scale), nl_move_start(nl_move_start), max_threads(max_threads) {
    // Constructor
    // Set some variables using the initialization list
    // Now set vector size
    steps_tested_displace_thread.resize(max_threads,vector<int>(8,0));
    steps_rejected_displace_thread.resize(max_threads,vector<int>(8,0));
    steps_tested_tether_thread.resize(max_threads,vector<int>(8,0));
    steps_rejected_tether_thread.resize(max_threads,vector<int>(8,0));
    steps_tested_mass_thread.resize(max_threads,vector<int>(8,0));
    steps_rejected_mass_thread.resize(max_threads,vector<int>(8,0));
    steps_tested_protein_thread.resize(max_threads,vector<int>(8,0));
    steps_rejected_protein_thread.resize(max_threads,vector<int>(8,0));
}

MCMoves::~MCMoves() {
    // Destructor
    // Does nothing
}

void MCMoves::DisplaceStep(MembraneMC& sys, NeighborList& nl, int vertex_trial, int thread_id) {
    // Pick random site and translate node
    Saru& local_generator = sys.generators[thread_id];
    sys.radii_tri[vertex_trial][0] += lambda*local_generator.d(-1.0,1.0);
    sys.radii_tri[vertex_trial][1] += lambda*local_generator.d(-1.0,1.0);
    sys.radii_tri[vertex_trial][2] += lambda*sys.lengths[1]*local_generator.d(-1.0,1.0);
    // Apply PBC on particles
    sys.radii_tri[vertex_trial][0] -= round(sys.radii_tri[vertex_trial][0]);
    sys.radii_tri[vertex_trial][1] -= round(sys.radii_tri[vertex_trial][1]);
    double phi_diff = 0; 
    double phi_diff_bending = 0;
    double area_diff = 0.0;
	// Loop through neighbor lists
	// Compare versus looping through all for verification
	// Determine index of current location
	int index_x = int(sys.lengths[0]*(sys.radii_tri[vertex_trial][0]+0.5)/nl.box_x);
	int index_y = int(sys.lengths[1]*(sys.radii_tri[vertex_trial][1]+0.5)/nl.box_y);
	int index_z = int((sys.radii_tri[vertex_trial][2]+sys.lengths[2])/nl.box_z);
    if(index_x == nl.nl_x) {
        index_x -= 1;
    }
    if(index_y == nl.nl_y) {
        index_y -= 1;
    }
	int index = index_x + index_y*nl.nl_x + index_z*nl.nl_x*nl.nl_y;
	// Loop through neighboring boxes
	// Check to make sure not counting self case
	for(int i=0; i<nl.neighbors[index].size(); i++) {
		for(int j=0; j<nl.neighbor_list[nl.neighbors[index][i]].size(); j++) {
			// Check particle interactions
            if(vertex_trial != nl.neighbor_list[nl.neighbors[index][i]][j]) {
                double length_neighbor = util.LengthLink(sys,vertex_trial,nl.neighbor_list[nl.neighbors[index][i]][j]);
                if(length_neighbor < 1.0) {
                    phi_diff += pow(10,100);
                }
            }
		}
	}
    // Check to see if particle moved out of bound of checkerboard
    int index_checkerboard_x = floor((sys.lengths[0]*(sys.radii_tri[vertex_trial][0]+0.5)-nl.cell_center_x)/nl.box_x_checkerboard);
	int index_checkerboard_y = floor((sys.lengths[1]*(sys.radii_tri[vertex_trial][1]+0.5)-nl.cell_center_y)/nl.box_y_checkerboard);
    if(index_checkerboard_x == -1) {
        index_checkerboard_x += nl.checkerboard_x;
    }
    if(index_checkerboard_y == -1) {
        index_checkerboard_y += nl.checkerboard_y;
    }
    if(index_checkerboard_x == nl.checkerboard_x) {
        index_checkerboard_x -= 1;
    }
    if(index_checkerboard_y == nl.checkerboard_y) {
        index_checkerboard_y -= 1;
    }
    int index_checkerboard = index_checkerboard_x + index_checkerboard_y*nl.checkerboard_x;
    if(index_checkerboard != nl.checkerboard_index[vertex_trial]) {
        phi_diff += pow(10,100);
    }
    
    // Energy due to bending energy
    // Precompute acceptance/rejectance probabilities
    double chance = local_generator.d();
    double chance_factor = -sys.temp*log(chance);
    if(phi_diff < pow(10,10)) {
        // Energy due to surface area
        for(int i=0; i<sys.point_triangle_list[vertex_trial].size(); i++) {
            int j = sys.point_triangle_list[vertex_trial][i];
            util.AreaNode(sys,j);
        }
        // Evaluate curvature energy at node changed and neighboring nodes
        util.EnergyNode(sys,vertex_trial);
        phi_diff_bending += sys.phi_vertex[vertex_trial] - sys.phi_vertex_original[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            util.EnergyNode(sys,sys.point_neighbor_list[vertex_trial][i]);
            phi_diff_bending += sys.phi_vertex[sys.point_neighbor_list[vertex_trial][i]] - sys.phi_vertex_original[sys.point_neighbor_list[vertex_trial][i]];
        }
        phi_diff += phi_diff_bending;
        // Evaluate energy due to surface tension
        area_diff += sys.sigma_vertex[vertex_trial]-sys.sigma_vertex_original[vertex_trial];
        phi_diff += sys.gamma_surf[sys.ising_array[vertex_trial]]*(sys.sigma_vertex[vertex_trial]-sys.sigma_vertex_original[vertex_trial]);
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            area_diff += sys.sigma_vertex[sys.point_neighbor_list[vertex_trial][i]]-sys.sigma_vertex_original[sys.point_neighbor_list[vertex_trial][i]];
            phi_diff += sys.gamma_surf[sys.ising_array[sys.point_neighbor_list[vertex_trial][i]]]*(sys.sigma_vertex[sys.point_neighbor_list[vertex_trial][i]]-sys.sigma_vertex_original[sys.point_neighbor_list[vertex_trial][i]]);
        }
    }
    // Run probabilities
    if(((chance_factor>phi_diff) && (phi_diff < pow(10,10)))) {
        // Accept move
        // Accept all trial values with for loop
        sys.phi_diff_thread[thread_id][0] += phi_diff;
        sys.phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys.area_diff_thread[thread_id][0] += area_diff;

        sys.radii_tri_original[vertex_trial] = sys.radii_tri[vertex_trial];        

        sys.phi_vertex_original[vertex_trial] = sys.phi_vertex[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            sys.phi_vertex_original[sys.point_neighbor_list[vertex_trial][i]] = sys.phi_vertex[sys.point_neighbor_list[vertex_trial][i]];
        }

        sys.mean_curvature_vertex_original[vertex_trial] = sys.mean_curvature_vertex[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            sys.mean_curvature_vertex_original[sys.point_neighbor_list[vertex_trial][i]] = sys.mean_curvature_vertex[sys.point_neighbor_list[vertex_trial][i]];
        }

        sys.sigma_vertex_original[vertex_trial] = sys.sigma_vertex[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            sys.sigma_vertex_original[sys.point_neighbor_list[vertex_trial][i]] = sys.sigma_vertex[sys.point_neighbor_list[vertex_trial][i]];
        }

        for(int i=0; i<sys.point_triangle_list[vertex_trial].size(); i++) {
            int j = sys.point_triangle_list[vertex_trial][i];
            sys.area_faces_original[j] = sys.area_faces[j];
        }

		// Change neighbor list if new index doesn't match up with old
		if(nl.neighbor_list_index[vertex_trial] != index) {
			// Determine which entry vertex trial was in original index bin and delete
			for(int i=0; i<nl.neighbor_list[nl.neighbor_list_index[vertex_trial]].size(); i++) {
				if(nl.neighbor_list[nl.neighbor_list_index[vertex_trial]][i] == vertex_trial) {
					nl.neighbor_list[nl.neighbor_list_index[vertex_trial]][i] = nl.neighbor_list[nl.neighbor_list_index[vertex_trial]].back();
					nl.neighbor_list[nl.neighbor_list_index[vertex_trial]].pop_back();
					i += nl.neighbor_list[nl.neighbor_list_index[vertex_trial]].size()+10;
				}
			}
			// Add to new bin
			nl.neighbor_list[index].push_back(vertex_trial);
			nl.neighbor_list_index[vertex_trial] = index;
		}
    }
    else {
        steps_rejected_displace_thread[thread_id][0] += 1;

        sys.radii_tri[vertex_trial] = sys.radii_tri_original[vertex_trial];

        sys.phi_vertex[vertex_trial] = sys.phi_vertex_original[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            sys.phi_vertex[sys.point_neighbor_list[vertex_trial][i]] = sys.phi_vertex_original[sys.point_neighbor_list[vertex_trial][i]];
        }

        sys.mean_curvature_vertex[vertex_trial] = sys.mean_curvature_vertex_original[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            sys.mean_curvature_vertex[sys.point_neighbor_list[vertex_trial][i]] = sys.mean_curvature_vertex_original[sys.point_neighbor_list[vertex_trial][i]];
        }

        sys.sigma_vertex[vertex_trial] = sys.sigma_vertex_original[vertex_trial];
        for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
            sys.sigma_vertex[sys.point_neighbor_list[vertex_trial][i]] = sys.sigma_vertex_original[sys.point_neighbor_list[vertex_trial][i]];
        }

        for(int i=0; i<sys.point_triangle_list[vertex_trial].size(); i++) {
            int j = sys.point_triangle_list[vertex_trial][i];
            sys.area_faces[j] = sys.area_faces_original[j];
        }
    }
    steps_tested_displace_thread[thread_id][0] += 1;
}

void MCMoves::TetherCut(MembraneMC& sys, NeighborList& nl, int vertex_trial, int thread_id) {
    // Choose link at random, destroy it, and create new link joining other ends of triangle
    // Have to update entries of sys.triangle_list, sys.point_neighbor_list, sys.point_neighbor_triangle, sys.point_triangle_list for this
    // Select random vertex
    Saru& local_generator = sys.generators[thread_id];
	// Reject move if acting on rollers
    // int vertex_trial = 1000;
    // Select random link from avaliable
    int link_trial = local_generator.rand_select(sys.point_neighbor_list[vertex_trial].size()-1);
    int vertex_trial_opposite = sys.point_neighbor_list[vertex_trial][link_trial];

    int triangle_trial[2]; 
    int point_trial[2]; 

    // Find the triangles to be changed using sys.point_neighbor_list
    triangle_trial[0] = sys.point_neighbor_triangle[vertex_trial][link_trial][0];
    triangle_trial[1] = sys.point_neighbor_triangle[vertex_trial][link_trial][1];

    // Find the two other points in the triangles using faces
    for(int i=0; i<2; i++) {
        if ((vertex_trial == sys.triangle_list[triangle_trial[i]][0]) || (vertex_trial_opposite == sys.triangle_list[triangle_trial[i]][0])) {
            if ((vertex_trial == sys.triangle_list[triangle_trial[i]][1]) || (vertex_trial_opposite == sys.triangle_list[triangle_trial[i]][1])) {
                point_trial[i] = sys.triangle_list[triangle_trial[i]][2];
            }
            else {
                point_trial[i] = sys.triangle_list[triangle_trial[i]][1];
            }
        }
        else {
            point_trial[i] = sys.triangle_list[triangle_trial[i]][0];
        }
    }
    if((vertex_trial < 0) || (vertex_trial_opposite < 0) || (point_trial[0] < 0) || (point_trial[1] < 0) || (vertex_trial_opposite > sys.vertices) || (point_trial[0] > sys.vertices) || (point_trial[1] > sys.vertices)) {
		steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if point_trial[0] and point_trial[1] are already linked
    for(int i=0; i<sys.point_neighbor_list[point_trial[0]].size(); i++) {
        if(sys.point_neighbor_list[point_trial[0]][i] == point_trial[1]) {
            steps_rejected_tether_thread[thread_id][0] += 1;
            steps_tested_tether_thread[thread_id][0] += 1;
            return;
        }
    }
    
	// Check to see if the limits for maximum or minimum number of nl.neighbors is exceeded
	if(((sys.point_neighbor_list[vertex_trial].size()-1) == sys.neighbor_min) || ((sys.point_neighbor_list[vertex_trial_opposite].size()-1) == sys.neighbor_min) || ((sys.point_neighbor_list[point_trial[0]].size()+1) == sys.neighbor_max) || ((sys.point_neighbor_list[point_trial[1]].size()+1) == sys.neighbor_max)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
		return;
	}
    
    // Check to see if any of the points are outside of the checkerboard set that vertex_trial is in
    if((nl.checkerboard_index[vertex_trial] != nl.checkerboard_index[vertex_trial_opposite]) || (nl.checkerboard_index[vertex_trial] != nl.checkerboard_index[point_trial[0]]) || (nl.checkerboard_index[vertex_trial] != nl.checkerboard_index[point_trial[1]])) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
		return;
    }

    // Check to see if trial points are too far apart
    double distance_point_trial = util.LengthLink(sys,point_trial[0], point_trial[1]); 
    if ((distance_point_trial > 1.673) || (distance_point_trial < 1.00)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        return;    
    }

    // Calculate detailed balance factor before everything changes
    // Basically, we now acc(v -> w) = gen(w->v) P(w)/gen(v->w) P(v)
    // Probability of gen(v->w) is 1/N*(1/nl.neighbors at vertex trial + 1/nl.neighbors at vertex_trial_opposite)
    // Similar for gen(w->v) except with point trial
    double db_factor = (1.0/(double(sys.point_neighbor_list[point_trial[0]].size())+1.0)+1.0/(double(sys.point_neighbor_list[point_trial[1]].size())+1.0))/(1.0/double(sys.point_neighbor_list[vertex_trial].size())+1.0/double(sys.point_neighbor_list[vertex_trial_opposite].size()));

    // Have all points needed, now just time to change sys.triangle_list, sys.point_neighbor_list, sys.point_neighbor_triangle, sys.point_triangle_list
    // Change triangle_list
    // Check orientation to see if consistent with before
    for(int i=0; i<3; i++) {
        if(sys.triangle_list[triangle_trial[0]][i] == vertex_trial_opposite) {
            sys.triangle_list[triangle_trial[0]][i] = point_trial[1];
            break;
        }
    }
    for(int i=0; i<3; i++) {
        if(sys.triangle_list[triangle_trial[1]][i] == vertex_trial) {
            sys.triangle_list[triangle_trial[1]][i] = point_trial[0];
            break;
        }
    }

    // Change sys.point_neighbor_list and sys.point_neighbor_triangle
    // Delete points
    int placeholder_nl = 0;
    while(placeholder_nl < sys.point_neighbor_list[vertex_trial].size()) {
        if(sys.point_neighbor_list[vertex_trial][placeholder_nl] == vertex_trial_opposite) {
            sys.point_neighbor_list[vertex_trial][placeholder_nl] = sys.point_neighbor_list[vertex_trial].back();
            sys.point_neighbor_triangle[vertex_trial][placeholder_nl] = sys.point_neighbor_triangle[vertex_trial].back();
            sys.point_neighbor_list[vertex_trial].pop_back();
            sys.point_neighbor_triangle[vertex_trial].pop_back();
            placeholder_nl = sys.neighbor_max;
        }
        placeholder_nl += 1;
    }
    placeholder_nl = 0;
    while(placeholder_nl < sys.point_neighbor_list[vertex_trial_opposite].size()) {
        if(sys.point_neighbor_list[vertex_trial_opposite][placeholder_nl] == vertex_trial) {
            sys.point_neighbor_list[vertex_trial_opposite][placeholder_nl] = sys.point_neighbor_list[vertex_trial_opposite].back();
            sys.point_neighbor_triangle[vertex_trial_opposite][placeholder_nl] = sys.point_neighbor_triangle[vertex_trial_opposite].back();
            sys.point_neighbor_list[vertex_trial_opposite].pop_back();
            sys.point_neighbor_triangle[vertex_trial_opposite].pop_back();
            placeholder_nl = sys.neighbor_max;
        }
        placeholder_nl += 1;
    }
    // Add points
    vector<int> points_add{triangle_trial[0],triangle_trial[1]};
    sys.point_neighbor_list[point_trial[0]].push_back(point_trial[1]);
    sys.point_neighbor_list[point_trial[1]].push_back(point_trial[0]);
    sys.point_neighbor_triangle[point_trial[0]].push_back(points_add);
    sys.point_neighbor_triangle[point_trial[1]].push_back(points_add);

    // Note that the definition of triangle_trial[0] and triangle_trial[1] have changed
    // Need to modify sys.point_neighbor_triangle entries between vertex_trial and point_trial[1]
    // and vertex_trial_opposite and point_trial[0] to swap triangle_trial[1] to triangle_trial[0]
    // and triangle_trial[0] to triangle_trial[1] respectively
    // Placeholder values so places needed are saved
    // vertex_trial and point_trial[1]
    for(int i=0; i<sys.point_neighbor_list[vertex_trial].size(); i++) {
        if(sys.point_neighbor_list[vertex_trial][i] == point_trial[1]) {
            if(sys.point_neighbor_triangle[vertex_trial][i][0] == triangle_trial[1]) {
                sys.point_neighbor_triangle[vertex_trial][i][0] = triangle_trial[0];
            }
            else if(sys.point_neighbor_triangle[vertex_trial][i][1] == triangle_trial[1]) {
                sys.point_neighbor_triangle[vertex_trial][i][1] = triangle_trial[0];
            }
        }
    }
    for(int i=0; i<sys.point_neighbor_list[point_trial[1]].size(); i++) {
        if(sys.point_neighbor_list[point_trial[1]][i] == vertex_trial) {
            if(sys.point_neighbor_triangle[point_trial[1]][i][0] == triangle_trial[1]) {
                sys.point_neighbor_triangle[point_trial[1]][i][0] = triangle_trial[0];
            }
            else if(sys.point_neighbor_triangle[point_trial[1]][i][1] == triangle_trial[1]) {
                sys.point_neighbor_triangle[point_trial[1]][i][1] = triangle_trial[0];
            }
        }
    }
    // vertex_trial_opposite and point_trial[0]
    for(int i=0; i<sys.point_neighbor_list[vertex_trial_opposite].size(); i++) {
        if(sys.point_neighbor_list[vertex_trial_opposite][i] == point_trial[0]) {
            if(sys.point_neighbor_triangle[vertex_trial_opposite][i][0] == triangle_trial[0]) {
                sys.point_neighbor_triangle[vertex_trial_opposite][i][0] = triangle_trial[1];
            }
            else if(sys.point_neighbor_triangle[vertex_trial_opposite][i][1] == triangle_trial[0]) {
                sys.point_neighbor_triangle[vertex_trial_opposite][i][1] = triangle_trial[1];
            }
        }
    }
    for(int i=0; i<sys.point_neighbor_list[point_trial[0]].size(); i++) {
        if(sys.point_neighbor_list[point_trial[0]][i] == vertex_trial_opposite) {
            if(sys.point_neighbor_triangle[point_trial[0]][i][0] == triangle_trial[0]) {
                sys.point_neighbor_triangle[point_trial[0]][i][0] = triangle_trial[1];
            }
            else if(sys.point_neighbor_triangle[point_trial[0]][i][1] == triangle_trial[0]) {
                sys.point_neighbor_triangle[point_trial[0]][i][1] = triangle_trial[1];
            }
        }
    }

    // Change sys.point_triangle_list
    placeholder_nl = 0;
    while(placeholder_nl < sys.point_triangle_list[vertex_trial].size()) {
        if(sys.point_triangle_list[vertex_trial][placeholder_nl] == triangle_trial[1]) {
            sys.point_triangle_list[vertex_trial][placeholder_nl] = sys.point_triangle_list[vertex_trial].back();
            sys.point_triangle_list[vertex_trial].pop_back();
            placeholder_nl = sys.neighbor_max;
        }
        placeholder_nl += 1;
    }
    placeholder_nl = 0;
    while(placeholder_nl < sys.point_triangle_list[vertex_trial_opposite].size()) {
        if(sys.point_triangle_list[vertex_trial_opposite][placeholder_nl] == triangle_trial[0]) {
            sys.point_triangle_list[vertex_trial_opposite][placeholder_nl] = sys.point_triangle_list[vertex_trial_opposite].back();
            sys.point_triangle_list[vertex_trial_opposite].pop_back();
            placeholder_nl = sys.neighbor_max;
        }
        placeholder_nl += 1;
    }
    // Add points
    sys.point_triangle_list[point_trial[0]].push_back(triangle_trial[1]);
    sys.point_triangle_list[point_trial[1]].push_back(triangle_trial[0]);

    // Evaluate energy difference
    // Evaluated at four nodes from two triangles changed
    double phi_diff = 0; 
    double phi_diff_bending = 0;
    double phi_diff_phi = 0;
    // Energy due to mean curvature
    util.EnergyNode(sys,vertex_trial);
    util.EnergyNode(sys,vertex_trial_opposite);
    util.EnergyNode(sys,point_trial[0]);
    util.EnergyNode(sys,point_trial[1]);

    phi_diff_bending += sys.phi_vertex[vertex_trial] - sys.phi_vertex_original[vertex_trial];
    phi_diff += sys.gamma_surf[sys.ising_array[vertex_trial]]*(sys.sigma_vertex[vertex_trial]-sys.sigma_vertex_original[vertex_trial]);

    phi_diff_bending += sys.phi_vertex[vertex_trial_opposite] - sys.phi_vertex_original[vertex_trial_opposite];
    phi_diff += sys.gamma_surf[sys.ising_array[vertex_trial_opposite]]*(sys.sigma_vertex[vertex_trial_opposite]-sys.sigma_vertex_original[vertex_trial_opposite]);

    phi_diff_bending += sys.phi_vertex[point_trial[0]] - sys.phi_vertex_original[point_trial[0]];
    phi_diff += sys.gamma_surf[sys.ising_array[point_trial[0]]]*(sys.sigma_vertex[point_trial[0]]-sys.sigma_vertex_original[point_trial[0]]);

    phi_diff_bending += sys.phi_vertex[point_trial[1]] - sys.phi_vertex_original[point_trial[1]];
    phi_diff += sys.gamma_surf[sys.ising_array[point_trial[1]]]*(sys.sigma_vertex[point_trial[1]]-sys.sigma_vertex_original[point_trial[1]]);

    phi_diff += phi_diff_bending;

    // Area change of triangles
    util.AreaNode(sys,triangle_trial[0]);
    util.AreaNode(sys,triangle_trial[1]);

    // Energy due to different Ising interactions
	phi_diff_phi += sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[vertex_trial_opposite]]*sys.ising_values[sys.ising_array[vertex_trial]]*sys.ising_values[sys.ising_array[vertex_trial_opposite]];
	phi_diff_phi -= sys.j_coupling[sys.ising_array[point_trial[0]]][sys.ising_array[point_trial[1]]]*sys.ising_values[sys.ising_array[point_trial[0]]]*sys.ising_values[sys.ising_array[point_trial[1]]];

    phi_diff += phi_diff_phi;
    double chance = local_generator.d();
    double chance_factor = -sys.temp*log(chance/db_factor);

    if((chance_factor>phi_diff) && (phi_diff < pow(10,10))) {
        // Accept move
        // New way that uses energy loop
        // Accept all trial values with for loop
        sys.phi_diff_thread[thread_id][0] += phi_diff;
        sys.phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys.phi_phi_diff_thread[thread_id][0] += phi_diff_phi;

        // Update original values
        // Have all points needed, now just time to change sys.triangle_list, sys.point_neighbor_list, link_triangle_list, sys.point_triangle_list 
        // Change triangle_list
        sys.triangle_list_original[triangle_trial[0]] = sys.triangle_list[triangle_trial[0]];
        sys.triangle_list_original[triangle_trial[1]] = sys.triangle_list[triangle_trial[1]];

        // Change sys.point_neighbor_list
        // Delete points
        sys.point_neighbor_list_original[vertex_trial] = sys.point_neighbor_list[vertex_trial];
        sys.point_neighbor_triangle_original[vertex_trial] = sys.point_neighbor_triangle[vertex_trial];
        sys.point_neighbor_list_original[vertex_trial_opposite] = sys.point_neighbor_list[vertex_trial_opposite];
        sys.point_neighbor_triangle_original[vertex_trial_opposite] = sys.point_neighbor_triangle[vertex_trial_opposite];

        // Add points
        sys.point_neighbor_list_original[point_trial[0]] = sys.point_neighbor_list[point_trial[0]];
        sys.point_neighbor_triangle_original[point_trial[0]] = sys.point_neighbor_triangle[point_trial[0]];
        sys.point_neighbor_list_original[point_trial[1]] = sys.point_neighbor_list[point_trial[1]];
        sys.point_neighbor_triangle_original[point_trial[1]] = sys.point_neighbor_triangle[point_trial[1]];

        // Change sys.point_triangle_list
        sys.point_triangle_list_original[vertex_trial] = sys.point_triangle_list[vertex_trial];
        sys.point_triangle_list_original[vertex_trial_opposite] = sys.point_triangle_list[vertex_trial_opposite];
        sys.point_triangle_list_original[point_trial[0]] = sys.point_triangle_list[point_trial[0]];
        sys.point_triangle_list_original[point_trial[1]] = sys.point_triangle_list[point_trial[1]];

        // Update energy values
        sys.phi_vertex_original[vertex_trial] = sys.phi_vertex[vertex_trial];
        sys.phi_vertex_original[vertex_trial_opposite] = sys.phi_vertex[vertex_trial_opposite];
        sys.phi_vertex_original[point_trial[0]] = sys.phi_vertex[point_trial[0]];
        sys.phi_vertex_original[point_trial[1]] = sys.phi_vertex[point_trial[1]];
        // Update mean curvature
        sys.mean_curvature_vertex_original[vertex_trial] = sys.mean_curvature_vertex[vertex_trial];
        sys.mean_curvature_vertex_original[vertex_trial_opposite] = sys.mean_curvature_vertex[vertex_trial_opposite];
        sys.mean_curvature_vertex_original[point_trial[0]] = sys.mean_curvature_vertex[point_trial[0]];
        sys.mean_curvature_vertex_original[point_trial[1]] = sys.mean_curvature_vertex[point_trial[1]];
        // Update sigma values
        sys.sigma_vertex_original[vertex_trial] = sys.sigma_vertex[vertex_trial];
        sys.sigma_vertex_original[vertex_trial_opposite] = sys.sigma_vertex[vertex_trial_opposite];
        sys.sigma_vertex_original[point_trial[0]] = sys.sigma_vertex[point_trial[0]];
        sys.sigma_vertex_original[point_trial[1]] = sys.sigma_vertex[point_trial[1]];

        // Update area of phases
        double area_total_diff = 0.0;
        area_total_diff += sys.area_faces[triangle_trial[0]] - sys.area_faces_original[triangle_trial[0]];
        area_total_diff += sys.area_faces[triangle_trial[1]] - sys.area_faces_original[triangle_trial[1]];
        sys.area_diff_thread[thread_id][0] += area_total_diff;

        sys.area_faces_original[triangle_trial[0]] = sys.area_faces[triangle_trial[0]];
        sys.area_faces_original[triangle_trial[1]] = sys.area_faces[triangle_trial[1]]; 
            
    }
    else {
        // cout << "Reject move at " << vertex_trial << endl;
        steps_rejected_tether_thread[thread_id][0] += 1;
        // Change new values to original values
        // Have all points needed, now just time to change sys.triangle_list, sys.point_neighbor_list, link_triangle_list, sys.point_triangle_list
        // Change sys.triangle_list
        sys.triangle_list[triangle_trial[0]] = sys.triangle_list_original[triangle_trial[0]];
        sys.triangle_list[triangle_trial[1]] = sys.triangle_list_original[triangle_trial[1]];

        // Change sys.point_neighbor_list
        // Delete points
        sys.point_neighbor_list[vertex_trial] = sys.point_neighbor_list_original[vertex_trial];
        sys.point_neighbor_triangle[vertex_trial] = sys.point_neighbor_triangle_original[vertex_trial];
        sys.point_neighbor_list[vertex_trial_opposite] = sys.point_neighbor_list_original[vertex_trial_opposite];
        sys.point_neighbor_triangle[vertex_trial_opposite] = sys.point_neighbor_triangle_original[vertex_trial_opposite];

        // Add points
        sys.point_neighbor_list[point_trial[0]] = sys.point_neighbor_list_original[point_trial[0]];
        sys.point_neighbor_triangle[point_trial[0]] = sys.point_neighbor_triangle_original[point_trial[0]];
        sys.point_neighbor_list[point_trial[1]] = sys.point_neighbor_list_original[point_trial[1]];
        sys.point_neighbor_triangle[point_trial[1]] = sys.point_neighbor_triangle_original[point_trial[1]];

        // Change sys.point_triangle_list
        sys.point_triangle_list[vertex_trial] = sys.point_triangle_list_original[vertex_trial];
        sys.point_triangle_list[vertex_trial_opposite] = sys.point_triangle_list_original[vertex_trial_opposite];
        sys.point_triangle_list[point_trial[0]] = sys.point_triangle_list_original[point_trial[0]];
        sys.point_triangle_list[point_trial[1]] = sys.point_triangle_list_original[point_trial[1]];

        // Update energy values
        sys.phi_vertex[vertex_trial] = sys.phi_vertex_original[vertex_trial];
        sys.phi_vertex[vertex_trial_opposite] = sys.phi_vertex_original[vertex_trial_opposite];
        sys.phi_vertex[point_trial[0]] = sys.phi_vertex_original[point_trial[0]];
        sys.phi_vertex[point_trial[1]] = sys.phi_vertex_original[point_trial[1]];
        // Update mean curvature
        sys.mean_curvature_vertex[vertex_trial] = sys.mean_curvature_vertex_original[vertex_trial];
        sys.mean_curvature_vertex[vertex_trial_opposite] = sys.mean_curvature_vertex_original[vertex_trial_opposite];
        sys.mean_curvature_vertex[point_trial[0]] = sys.mean_curvature_vertex_original[point_trial[0]];
        sys.mean_curvature_vertex[point_trial[1]] = sys.mean_curvature_vertex_original[point_trial[1]];
        // Update sigma values
        sys.sigma_vertex[vertex_trial] = sys.sigma_vertex_original[vertex_trial];
        sys.sigma_vertex[vertex_trial_opposite] = sys.sigma_vertex_original[vertex_trial_opposite];
        sys.sigma_vertex[point_trial[0]] = sys.sigma_vertex_original[point_trial[0]];
        sys.sigma_vertex[point_trial[1]] = sys.sigma_vertex_original[point_trial[1]];
   
        // Update area of phases
        sys.area_faces[triangle_trial[0]] = sys.area_faces_original[triangle_trial[0]];
        sys.area_faces[triangle_trial[1]] = sys.area_faces_original[triangle_trial[1]]; 

        
    }
    steps_tested_tether_thread[thread_id][0] += 1;
}

void MCMoves::ChangeMassNonCon(MembraneMC& sys, NeighborList& nl, int vertex_trial, int thread_id) {
    // Pick random site and change the Ising array value
    // Non-mass conserving
    Saru& local_generator = sys.generators[thread_id];
    // Have to have implementation that chooses from within the same checkerboard set
    if(sys.ising_array[vertex_trial] == 2) {
        // If protein node is selected, do a MoveProteinGen move instead
        MoveProteinGen(sys, nl, vertex_trial, thread_id);
        return;
	} 
    // Change spin
    int ising_array_trial = 0;
    if(sys.ising_array[vertex_trial] == 0){
        ising_array_trial = 1;
    }

    double phi_diff_phi = 0;

    for(int j=0; j<sys.point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[sys.ising_array[vertex_trial]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[ising_array_trial];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    double phi_diff_mag = -sys.h_external*(sys.ising_values[ising_array_trial]-sys.ising_values[sys.ising_array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = sys.mean_curvature_vertex[vertex_trial]-sys.spon_curv[ising_array_trial];
    sys.phi_vertex[vertex_trial] = sys.k_b[ising_array_trial]*sys.sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double phi_diff_bending = sys.phi_vertex[vertex_trial] - sys.phi_vertex_original[vertex_trial];
    double phi_diff = phi_diff_phi+phi_diff_bending+phi_diff_mag;
    phi_diff += (sys.gamma_surf[ising_array_trial]-sys.gamma_surf[sys.ising_array[vertex_trial]])*sys.sigma_vertex[vertex_trial];
    double chance = local_generator.d();
    if(chance<exp(-phi_diff/sys.temp)){
        sys.phi_diff_thread[thread_id][0] += phi_diff;
        sys.phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys.phi_phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys.mass_diff_thread[thread_id][0] += (ising_array_trial-sys.ising_array[vertex_trial]);
        sys.magnet_diff_thread[thread_id][0] += (sys.ising_values[ising_array_trial]-sys.ising_values[sys.ising_array[vertex_trial]]);
        sys.ising_array[vertex_trial] = ising_array_trial;
        sys.phi_vertex_original[vertex_trial] = sys.phi_vertex[vertex_trial];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        sys.phi_vertex[vertex_trial] = sys.phi_vertex_original[vertex_trial];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;
}

void MCMoves::ChangeMassCon(MembraneMC& sys, NeighborList& nl, int vertex_trial, int thread_id) {
    // Pick random site and attemp to swap Ising array value with array in random nearest neighbor direction
    // Mass conserving
    Saru& local_generator = sys.generators[thread_id];
    if(sys.ising_array[vertex_trial] == 2) {
        // If protein node is selected, do a MoveProteinGen move instead
        MoveProteinGen(sys, nl, vertex_trial, thread_id);
        return;
	} 
    // Pick random direction
    int link_trial = local_generator.rand_select(sys.point_neighbor_list[vertex_trial].size()-1);
    int vertex_trial_opposite = sys.point_neighbor_list[vertex_trial][link_trial];

    // For now reject if the neighboring sites have the same array value or protein type
    if(sys.ising_array[vertex_trial] == sys.ising_array[vertex_trial_opposite]) {
        steps_rejected_mass_thread[thread_id][0] += 1;
        steps_tested_mass_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if vertex_trial_opposite is not in same checkerboard set
    if(nl.checkerboard_index[vertex_trial] != nl.checkerboard_index[vertex_trial_opposite]) {
        steps_rejected_mass_thread[thread_id][0] += 1;
    	steps_tested_mass_thread[thread_id][0] += 1;
		return;
    }

    // Set trial values
    int ising_array_trial_1 = sys.ising_array[vertex_trial_opposite];
    int ising_array_trial_2 = sys.ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_diff_phi = 0;
    for(int j=0; j<sys.point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[sys.ising_array[vertex_trial]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial_1][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[ising_array_trial_1];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys.mean_curvature_vertex[vertex_trial]-sys.spon_curv[ising_array_trial_1];
    sys.phi_vertex[vertex_trial] = sys.k_b[ising_array_trial_1]*sys.sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<sys.point_neighbor_list[vertex_trial_opposite].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[vertex_trial_opposite]][sys.ising_array[sys.point_neighbor_list[vertex_trial_opposite][j]]]*sys.ising_values[sys.ising_array[vertex_trial_opposite]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial_2][sys.ising_array[sys.point_neighbor_list[vertex_trial_opposite][j]]]*sys.ising_values[ising_array_trial_2];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[vertex_trial_opposite][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys.mean_curvature_vertex[vertex_trial_opposite]-sys.spon_curv[ising_array_trial_2];
    sys.phi_vertex[vertex_trial_opposite] = sys.k_b[ising_array_trial_2]*sys.sigma_vertex[vertex_trial_opposite]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    phi_diff_phi += (sys.j_coupling[ising_array_trial_1][sys.ising_array[vertex_trial_opposite]]*sys.ising_values[ising_array_trial_1]-sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[vertex_trial_opposite]]*sys.ising_values[sys.ising_array[vertex_trial]])*sys.ising_values[sys.ising_array[vertex_trial_opposite]];
    phi_diff_phi += (sys.j_coupling[ising_array_trial_2][sys.ising_array[vertex_trial]]*sys.ising_values[ising_array_trial_2]-sys.j_coupling[sys.ising_array[vertex_trial_opposite]][sys.ising_array[vertex_trial]]*sys.ising_values[sys.ising_array[vertex_trial_opposite]])*sys.ising_values[sys.ising_array[vertex_trial]];

    double phi_diff_bending = sys.phi_vertex[vertex_trial] - sys.phi_vertex_original[vertex_trial];
    phi_diff_bending += sys.phi_vertex[vertex_trial_opposite] - sys.phi_vertex_original[vertex_trial_opposite];
    double phi_diff = phi_diff_bending+phi_diff_phi;
    phi_diff += (sys.gamma_surf[ising_array_trial_1]-sys.gamma_surf[sys.ising_array[vertex_trial]])*sys.sigma_vertex[vertex_trial];
    phi_diff += (sys.gamma_surf[ising_array_trial_2]-sys.gamma_surf[sys.ising_array[vertex_trial_opposite]])*sys.sigma_vertex[vertex_trial_opposite];
    double chance = local_generator.d();
    if(chance<exp(-phi_diff/sys.temp)){
        sys.phi_diff_thread[thread_id][0] += phi_diff;
        sys.phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys.phi_phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys.ising_array[vertex_trial] = ising_array_trial_1;
        sys.ising_array[vertex_trial_opposite] = ising_array_trial_2;
        sys.phi_vertex_original[vertex_trial] = sys.phi_vertex[vertex_trial];
        sys.phi_vertex_original[vertex_trial_opposite] = sys.phi_vertex[vertex_trial_opposite];
    }
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        sys.phi_vertex[vertex_trial] = sys.phi_vertex_original[vertex_trial];
        sys.phi_vertex[vertex_trial_opposite] = sys.phi_vertex_original[vertex_trial_opposite];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;
}

void MCMoves::MoveProteinGen(MembraneMC& sys, NeighborList& nl, int vertex_trial, int thread_id) {
    // Pick random protein and attempt to move it in the y-direction
    // As protein's not merging, don't let them
    Saru& local_generator = sys.generators[thread_id];
    // Pick direction
    // Instead just go with one it's nl.neighbors
    int direction_trial = local_generator.rand_select(sys.point_neighbor_list[vertex_trial].size()-1);
    int center_trial = sys.point_neighbor_list[vertex_trial][direction_trial]; 
    // Reject if about to swap with another protein of the same type
    if((sys.protein_node[vertex_trial] != -1) && (sys.protein_node[vertex_trial] == sys.protein_node[center_trial])) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Reject if not in the same checkerboard set
    if(nl.checkerboard_index[vertex_trial] != nl.checkerboard_index[center_trial]) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Have to break down into cases
    int case_vertex = sys.protein_node[vertex_trial];
    int case_center = sys.protein_node[center_trial];

    // Energetics of swapping those two
    // Set trial values
    int ising_array_trial_1 = sys.ising_array[center_trial];
    int ising_array_trial_2 = sys.ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_diff_phi = 0;
    for(int j=0; j<sys.point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[sys.ising_array[vertex_trial]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial_1][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[ising_array_trial_1];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys.mean_curvature_vertex[vertex_trial]-sys.spon_curv[ising_array_trial_1];
    sys.phi_vertex[vertex_trial] = sys.k_b[ising_array_trial_1]*sys.sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<sys.point_neighbor_list[center_trial].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[center_trial]][sys.ising_array[sys.point_neighbor_list[center_trial][j]]]*sys.ising_values[sys.ising_array[center_trial]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial_2][sys.ising_array[sys.point_neighbor_list[center_trial][j]]]*sys.ising_values[ising_array_trial_2];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[center_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys.mean_curvature_vertex[center_trial]-sys.spon_curv[ising_array_trial_2];
    sys.phi_vertex[center_trial] = sys.k_b[ising_array_trial_2]*sys.sigma_vertex[center_trial]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    phi_diff_phi += (sys.j_coupling[ising_array_trial_1][sys.ising_array[center_trial]]*sys.ising_values[ising_array_trial_1]-sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[center_trial]]*sys.ising_values[sys.ising_array[vertex_trial]])*sys.ising_values[sys.ising_array[center_trial]];
    phi_diff_phi += (sys.j_coupling[ising_array_trial_2][sys.ising_array[vertex_trial]]*sys.ising_values[ising_array_trial_2]-sys.j_coupling[sys.ising_array[center_trial]][sys.ising_array[vertex_trial]]*sys.ising_values[sys.ising_array[center_trial]])*sys.ising_values[sys.ising_array[vertex_trial]];
    
    double phi_diff_bending = sys.phi_vertex[vertex_trial] - sys.phi_vertex_original[vertex_trial];
    phi_diff_bending += sys.phi_vertex[center_trial] - sys.phi_vertex_original[center_trial];
    double phi_diff = phi_diff_phi+phi_diff_bending;
    phi_diff += (sys.gamma_surf[ising_array_trial_1]-sys.gamma_surf[sys.ising_array[vertex_trial]])*sys.sigma_vertex[vertex_trial];
    phi_diff += (sys.gamma_surf[ising_array_trial_2]-sys.gamma_surf[sys.ising_array[center_trial]])*sys.sigma_vertex[center_trial];
    double chance = local_generator.d();
    double db_factor = double(sys.point_neighbor_list[vertex_trial].size())/double(sys.point_neighbor_list[center_trial].size());
    double chance_factor = -sys.temp*log(chance/db_factor);

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(chance_factor>phi_diff) {
        accept = true;
    }

    if(accept == true) {
        sys.phi_diff_thread[thread_id][0] += phi_diff;
        sys.phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys.phi_phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys.ising_array[vertex_trial] = ising_array_trial_1;
        sys.ising_array[center_trial] = ising_array_trial_2;
        sys.phi_vertex_original[vertex_trial] = sys.phi_vertex[vertex_trial];
        sys.phi_vertex_original[center_trial] = sys.phi_vertex[center_trial];
        sys.protein_node[vertex_trial] = case_center;
        sys.protein_node[center_trial] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        sys.phi_vertex[vertex_trial] = sys.phi_vertex_original[vertex_trial];
        sys.phi_vertex[center_trial] = sys.phi_vertex_original[center_trial];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void MCMoves::MoveProteinNL(MembraneMC& sys, int vertex_trial, int vertex_trial_2, int thread_id) {
    // Nonlocal movement of protein nodes using preselected vertex_trial and vertex_trial_2
    Saru& local_generator = sys.generators[thread_id];
    // Reject if about to swap with another protein of the same type
    if(((sys.protein_node[vertex_trial] == -1) || (sys.protein_node[vertex_trial_2] == -1)) && (sys.protein_node[vertex_trial] == sys.protein_node[vertex_trial_2])) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Have to break down into cases
    int case_vertex = sys.protein_node[vertex_trial];
    int case_center = sys.protein_node[vertex_trial_2];

    // Energetics of swapping those two
    // Set trial values
    int ising_array_trial_1 = sys.ising_array[vertex_trial_2];
    int ising_array_trial_2 = sys.ising_array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double phi_diff_phi = 0;
    for(int j=0; j<sys.point_neighbor_list[vertex_trial].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[vertex_trial]][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[sys.ising_array[vertex_trial]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial_1][sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]]*sys.ising_values[ising_array_trial_1];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = sys.mean_curvature_vertex[vertex_trial]-sys.spon_curv[ising_array_trial_1];
    sys.phi_vertex[vertex_trial] = sys.k_b[ising_array_trial_1]*sys.sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<sys.point_neighbor_list[vertex_trial_2].size(); j++) {
        double Site_diff = sys.j_coupling[sys.ising_array[vertex_trial_2]][sys.ising_array[sys.point_neighbor_list[vertex_trial_2][j]]]*sys.ising_values[sys.ising_array[vertex_trial_2]];
        double Site_diff_2 = sys.j_coupling[ising_array_trial_2][sys.ising_array[sys.point_neighbor_list[vertex_trial_2][j]]]*sys.ising_values[ising_array_trial_2];
        phi_diff_phi -= (Site_diff_2-Site_diff)*sys.ising_values[sys.ising_array[sys.point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = sys.mean_curvature_vertex[vertex_trial_2]-sys.spon_curv[ising_array_trial_2];
    sys.phi_vertex[vertex_trial_2] = sys.k_b[ising_array_trial_2]*sys.sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    double phi_diff_bending = sys.phi_vertex[vertex_trial] - sys.phi_vertex_original[vertex_trial];
    phi_diff_bending += sys.phi_vertex[vertex_trial_2] - sys.phi_vertex_original[vertex_trial_2];
    double phi_diff = phi_diff_phi+phi_diff_bending;
    phi_diff += (sys.gamma_surf[ising_array_trial_1]-sys.gamma_surf[sys.ising_array[vertex_trial]])*sys.sigma_vertex[vertex_trial];
    phi_diff += (sys.gamma_surf[ising_array_trial_2]-sys.gamma_surf[sys.ising_array[vertex_trial_2]])*sys.sigma_vertex[vertex_trial_2];
    double chance = local_generator.d();
    double chance_factor = -sys.temp*log(chance);

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(chance_factor>phi_diff) {
        accept = true;
    }
    if(accept == true) {
        sys.phi_diff_thread[thread_id][0] += phi_diff;
        sys.phi_bending_diff_thread[thread_id][0] += phi_diff_bending;
        sys.phi_phi_diff_thread[thread_id][0] += phi_diff_phi;
        sys.ising_array[vertex_trial] = ising_array_trial_1;
        sys.ising_array[vertex_trial_2] = ising_array_trial_2;
        sys.phi_vertex_original[vertex_trial] = sys.phi_vertex[vertex_trial];
        sys.phi_vertex_original[vertex_trial_2] = sys.phi_vertex[vertex_trial_2];
        sys.protein_node[vertex_trial] = case_center;
        sys.protein_node[vertex_trial_2] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        sys.phi_vertex[vertex_trial] = sys.phi_vertex_original[vertex_trial];
        sys.phi_vertex[vertex_trial_2] = sys.phi_vertex_original[vertex_trial_2];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void MCMoves::ChangeArea(MembraneMC& sys, NeighborList& nl) {
    // Attempt to modify the box size
    chrono::steady_clock::time_point t1_area;
    chrono::steady_clock::time_point t2_area;
    t1_area = chrono::steady_clock::now();

    double scale_xy_trial = sys.scale_xy+lambda_scale*sys.generator.d(-1.0,1.0);
    if(scale_xy_trial <= 0.0) {
        steps_rejected_area++;
        steps_tested_area++;
        return;
    }
    sys.lengths[0] = sys.lengths_base[0]*scale_xy_trial;
    sys.lengths[1] = sys.lengths_base[1]*scale_xy_trial;
    t2_area = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_area-t1_area;
    sys.time_storage_area[0] += time_span.count();
    // Reform neighbor list
    t1_area = chrono::steady_clock::now();
    nl.GenerateNeighborList(sys);
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    sys.time_storage_area[1] += time_span.count();
    // Store original values
    t1_area = chrono::steady_clock::now();
    double phi_ = sys.phi;
    double phi_bending_ = sys.phi_bending;
    double phi_phi_ = sys.phi_phi;
    double area_total_ = sys.area_total;
    // Recompute energy
    // Note that this version doesn't override all variables
    util.InitializeEnergyScale(sys,nl);
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    sys.time_storage_area[2] += time_span.count();
    // Now accept/reject
    t1_area = chrono::steady_clock::now();
    double chance = sys.generator.d();
    double phi_diff = (sys.phi-phi_)-sys.temp*2*sys.vertices*log(scale_xy_trial/sys.scale_xy);
    if((chance<exp(-phi_diff/sys.temp)) && (phi_diff < pow(10,10))) {
        sys.scale_xy = scale_xy_trial;
        sys.lengths_old = sys.lengths;
        nl.box_x = sys.lengths[0]/double(nl.nl_x); 
        nl.box_y = sys.lengths[1]/double(nl.nl_y); 
        #pragma omp parallel for
        for(int i=0; i<sys.vertices; i++) {
            sys.phi_vertex_original[i] = sys.phi_vertex[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys.faces; i++) {
            sys.area_faces_original[i] = sys.area_faces[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys.vertices; i++) {
            sys.mean_curvature_vertex_original[i] = sys.mean_curvature_vertex[i]; 
        }
        #pragma omp parallel for
        for(int i=0; i<sys.vertices; i++) {
            sys.sigma_vertex_original[i] = sys.sigma_vertex[i]; 
        }
    }
    else {
        steps_rejected_area++;
        sys.phi = phi_;
        sys.phi_bending = phi_bending_;
        sys.phi_phi = phi_phi_;
        sys.area_total = area_total_;
        sys.lengths = sys.lengths_old;
        nl.GenerateNeighborList(sys);
        nl.box_x = sys.lengths[0]/double(nl.nl_x); 
        nl.box_y = sys.lengths[1]/double(nl.nl_y); 
        #pragma omp parallel for
        for(int i=0; i<sys.vertices; i++) {
            sys.phi_vertex[i] = sys.phi_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys.faces; i++) {
            sys.area_faces[i] = sys.area_faces_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys.vertices; i++) {
            sys.mean_curvature_vertex[i] = sys.mean_curvature_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<sys.vertices; i++) {
            sys.sigma_vertex[i] = sys.sigma_vertex_original[i];
        }
    }
    steps_tested_area++;
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    sys.time_storage_area[3] += time_span.count();
}
