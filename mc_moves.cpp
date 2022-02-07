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

void MCMoves::DisplaceStep(int vertex_trial, int thread_id) {
    // Pick random site and translate node
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
	// cout << "Vertex trial " << vertex_trial << endl;
	// cout << "x " << Radius_x_tri[vertex_trial] << " y " << Radius_y_tri[vertex_trial] << " z " << Radius_z_tri[vertex_trial] << endl;
    // Trial move - change radius
    // Comment x,y out for z moves only
    Radius_x_tri[vertex_trial] += lambda*local_generator.d(-1.0,1.0);
    Radius_y_tri[vertex_trial] += lambda*local_generator.d(-1.0,1.0);
    Radius_z_tri[vertex_trial] += lambda*Length_y*local_generator.d(-1.0,1.0);
    // Apply PBC on particles
    Radius_x_tri[vertex_trial] = fmod(fmod(Radius_x_tri[vertex_trial],1.0)+1.0,1.0);
    Radius_y_tri[vertex_trial] = fmod(fmod(Radius_y_tri[vertex_trial],1.0)+1.0,1.0);
	// cout << "x " << Radius_x_tri[vertex_trial] << " y " << Radius_y_tri[vertex_trial] << " z " << Radius_z_tri[vertex_trial] << endl;
    double Phi_diff = 0; 
    double Phi_diff_bending = 0;
	// Loop through neighbor lists
	// Compare versus looping through all for verification
	// Determine index of current location
	int index_x = int(Length_x*Radius_x_tri[vertex_trial]/box_x);
	int index_y = int(Length_y*Radius_y_tri[vertex_trial]/box_y);
	int index_z = int((Radius_z_tri[vertex_trial]+Length_z)/box_z);
    if(index_x == nl_x) {
        index_x -= 1;
    }
    if(index_y == nl_y) {
        index_y -= 1;
    }
	int index = index_x + index_y*nl_x + index_z*nl_x*nl_y;
	// Loop through neighboring boxes
	// Check to make sure not counting self case
	/*
	cout << "At neighbor part!" << endl;
	cout << "Vertex trial" << endl;
	cout << "Index " << index << endl;
	*/
	// cout << "Displace step" << endl;
	for(int i=0; i<neighbors[index].size(); i++) {
		// cout << neighbors[index][i] << " ";
		// cout << "Number of particles in bin " << neighbor_list[neighbors[index][i]].size() << endl;
		// cout << "Particles time" << endl;
		for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
			// cout << neighbor_list[neighbors[index][i]][j] << " ";
			// Check particle interactions
            if(vertex_trial != neighbor_list[neighbors[index][i]][j]) {
                double length_neighbor = lengthLink(vertex_trial,neighbor_list[neighbors[index][i]][j]);
                // cout << "length from particle to particle " << length_neighbor << endl;
                if(length_neighbor < 1.0) {
                    // cout << "Neighbor " << neighbor_list[neighbors[index][i]][j] << " " << Radius_x_tri[neighbor_list[neighbors[index][i]][j]] << " " << Radius_y_tri[neighbor_list[neighbors[index][i]][j]] << " " << Radius_z_tri[neighbor_list[neighbors[index][i]][j]] << endl;
                    Phi_diff += pow(10,100);
                }
            }
		}
		// cout << endl;
	}
	// cout << "Displace check " << check << endl;
	// cout << "Phi_diff after neighbor " << Phi_diff << endl;
	// cout << "Probability " << exp(-Phi_diff/T) << endl;
    // Check to see if particle moved out of bound
    int index_checkerboard_x = floor((Length_x*Radius_x_tri[vertex_trial]-cell_center_x)/box_x_checkerboard);
	int index_checkerboard_y = floor((Length_y*Radius_y_tri[vertex_trial]-cell_center_y)/box_y_checkerboard);
    // cout << index_checkerboard_x << " " << index_checkerboard_y << endl;
    if(index_checkerboard_x == -1) {
        index_checkerboard_x += checkerboard_x;
    }
    if(index_checkerboard_y == -1) {
        index_checkerboard_y += checkerboard_y;
    }
    if(index_checkerboard_x == checkerboard_x) {
        index_checkerboard_x -= 1;
    }
    if(index_checkerboard_y == checkerboard_y) {
        index_checkerboard_y -= 1;
    }
    // cout << "After " << index_checkerboard_x << " " << index_checkerboard_y << endl;
    int index_checkerboard = index_checkerboard_x + index_checkerboard_y*checkerboard_x;
    if(index_checkerboard != checkerboard_index[vertex_trial]) {
        Phi_diff += pow(10,100);
    }
    
    // Energy due to mean curvature
	/*
    int max_local_neighbor = 0;
    for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
        max_local_neighbor += point_neighbor_max[point_neighbor_list[vertex_trial][i]];
    }
    int local_neighbor_list[max_local_neighbor];
    int counter = 0;
    for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
        for(int j=0; j<point_neighbor_max[point_neighbor_list[vertex_trial][i]]; j++) {
            local_neighbor_list[counter] = point_neighbor_list[point_neighbor_list[vertex_trial][i]][j];
            counter++;
        }     
    }
    // /
    for(int i=0; i<max_local_neighbor; i++) {
        cout << local_neighbor_list[i] << " ";
    }
    cout << endl;
    // /
    sort(local_neighbor_list,local_neighbor_list+max_local_neighbor);
    // /
    for(int i=0; i<max_local_neighbor; i++) {
        cout << local_neighbor_list[i] << " ";
    }
    cout << endl;
    // / 
    int max_unique = 0;
    int neighbor_unique_raw[max_local_neighbor];
    // Go through sorted array
    for(int i=0; i<max_local_neighbor; i++) {
        while ((i < (max_local_neighbor-1)) && (local_neighbor_list[i] == local_neighbor_list[i+1])) {
            i++;
        }
        neighbor_unique_raw[max_unique] = local_neighbor_list[i]; 
        max_unique++;
    }
    // /
    for(int i=0; i<max_unique; i++) {
        cout << neighbor_unique_raw[i] << " ";
    }    
    cout << endl;
    // /
    for(int i=0; i<max_unique; i++) {
        energyNode(neighbor_unique_raw[i]);
        Phi_diff += phi_vertex[neighbor_unique_raw[i]] - phi_vertex_original[neighbor_unique_raw[i]];
    }
	*/
    bool accept = false;
    double Chance = local_generator.d();
    double Chance_factor = -T*log(Chance);
    if(Phi_diff < pow(10,10)) {
        // Evaluate energy difference
        // Energy due to surface area
        for(int i=0; i<point_triangle_max[vertex_trial]; i++) {
            int j = point_triangle_list[vertex_trial][i];
            areaNode(j);
            Phi_diff += gamma_surf[0]*(area_faces[j] - area_faces_original[j]);
        }
        // Evaluated at node changed and neighboring nodes
        energyNode(vertex_trial);
        Phi_diff_bending += phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            energyNode(point_neighbor_list[vertex_trial][i]);
            Phi_diff_bending += phi_vertex[point_neighbor_list[vertex_trial][i]] - phi_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        Phi_diff += Phi_diff_bending;
        // cout << "Phi_diff after phi_vertexes " << Phi_diff << endl;
        // cout << "Phi_diff after checking boundaries " << Phi_diff << endl;
    }
    /*
    for(int i=0; i<vertices; i++) {
        energyNode(i); // Contribution due to mean curvature and surface area
        Phi_diff += phi_vertex[i] - phi_vertex_original[i];
    }
    */

    // EnergyStep();
    // double Phi_diff = Phi - Phi_original;

    // Actual area
    /*
    double total_area = 0.0;
    for(int i=0; i<faces; i++) {
        areaNode(i);
        total_area += area_faces[i];
    }
    cout << "Actual area is " << total_area << endl;
    */
	// Check to see if displacement violates actin hard sphere
	/*
	if((abs(Radius_x_tri[vertex_trial]-Length_x/2.0) < (1.0/5.0*Length_x+1.0)) && (abs(Radius_y_tri[vertex_trial]-Length_y/2.0) < (1.0/4.0*Length_y+1.0))) {
		for(int i = 0; i < particle_coord_x.size(); i++) {
			double dist_check = pow(pow(Radius_x_tri[vertex_trial]-particle_coord_x[i],2)+pow(Radius_y_tri[vertex_trial]-particle_coord_y[i],2)+pow(Radius_z_tri[vertex_trial]-particle_coord_z[i],2),0.5);
			if (dist_check < 1.09) {
				// cout << i << " " << dist_check << endl;
				Phi_diff += pow(10,100);
			}
		}
	}
	*/
	// cout << "Phi_diff before neighbor list " << Phi_diff << endl;
    if((accept == true) || ((Chance_factor>Phi_diff) && (Phi_diff < pow(10,10)))) {
        // Accept move
        // New way that uses energy loop
        // Accept all trial values with for loop
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;

        /*
        for(int i=0; i<vertices; i++) {
            phi_vertex_original[i] = phi_vertex[i];
        }
        */
		/*
        for(int i=0; i<max_unique; i++) {
            phi_vertex_original[neighbor_unique_raw[i]] = phi_vertex[neighbor_unique_raw[i]];
        }  
        */
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            phi_vertex_original[point_neighbor_list[vertex_trial][i]] = phi_vertex[point_neighbor_list[vertex_trial][i]];
        }
        mean_curvature_vertex_original[vertex_trial] = mean_curvature_vertex[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            mean_curvature_vertex_original[point_neighbor_list[vertex_trial][i]] = mean_curvature_vertex[point_neighbor_list[vertex_trial][i]];
        }
        sigma_vertex_original[vertex_trial] = sigma_vertex[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            sigma_vertex_original[point_neighbor_list[vertex_trial][i]] = sigma_vertex[point_neighbor_list[vertex_trial][i]];
        }
        Radius_x_tri_original[vertex_trial] = Radius_x_tri[vertex_trial];        
        Radius_y_tri_original[vertex_trial] = Radius_y_tri[vertex_trial];
        Radius_z_tri_original[vertex_trial] = Radius_z_tri[vertex_trial];
        double Area_total_diff_local = 0.0;
        for(int i=0; i<point_triangle_max[vertex_trial]; i++) {
            int j = point_triangle_list[vertex_trial][i];
            Area_total_diff_local += area_faces[j] - area_faces_original[j];
            area_faces_original[j] = area_faces[j];
        }
        Area_diff_thread[thread_id][0] += Area_total_diff_local;
		// Change neighbor list if new index doesn't match up with old
		if(neighbor_list_index[vertex_trial] != index) {
			// Determine which entry vertex trial was in original index bin and delete
			// cout << "Deleting index!" << endl;
			// cout << "Old neighbor list and size " << neighbor_list[neighbor_list_index[vertex_trial]].size() <<  endl;
			/*
			for(int i=0; i<neighbor_list[neighbor_list_index[vertex_trial]].size(); i++) {
				cout << neighbor_list[neighbor_list_index[vertex_trial]][i] << " ";
			}
			*/
			// cout << endl;
			for(int i=0; i<neighbor_list[neighbor_list_index[vertex_trial]].size(); i++) {
				if(neighbor_list[neighbor_list_index[vertex_trial]][i] == vertex_trial) {
					neighbor_list[neighbor_list_index[vertex_trial]][i] = neighbor_list[neighbor_list_index[vertex_trial]].back();
					neighbor_list[neighbor_list_index[vertex_trial]].pop_back();
					i += neighbor_list[neighbor_list_index[vertex_trial]].size()+10;
				}
			}
			// cout << "Should have been deleted and size " << neighbor_list[neighbor_list_index[vertex_trial]].size() << endl;
			/*
			for(int i=0; i<neighbor_list[neighbor_list_index[vertex_trial]].size(); i++) {
				cout << neighbor_list[neighbor_list_index[vertex_trial]][i] << " ";
			}
			*/
			// cout << endl;
			// Add to new bin
			// Old neighbor list
			// cout << "Old neighbor list at new index" << endl;
			/*
			for(int i=0; i<neighbor_list[index].size(); i++) {
				cout << neighbor_list[index][i] << " ";
			}
			*/
			// cout << endl;
			neighbor_list[index].push_back(vertex_trial);
			neighbor_list_index[vertex_trial] = index;
			// cout << "New neighbor list at new index" << endl;
			/*
			for(int i=0; i<neighbor_list[index].size(); i++) {
				cout << neighbor_list[index][i] << " ";
			}
			*/
			// cout << endl;
		}
    }
    else {
        steps_rejected_displace_thread[thread_id][0] += 1;
        Radius_x_tri[vertex_trial] = Radius_x_tri_original[vertex_trial];
        Radius_y_tri[vertex_trial] = Radius_y_tri_original[vertex_trial];
        Radius_z_tri[vertex_trial] = Radius_z_tri_original[vertex_trial];

        /*
        for(int i=0; i<vertices; i++) {
            phi_vertex[i] = phi_vertex_original[i];
        }
        */
		/*
        for(int i=0; i<max_unique; i++) {
            phi_vertex[neighbor_unique_raw[i]] = phi_vertex_original[neighbor_unique_raw[i]];
        } 
		*/     
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            phi_vertex[point_neighbor_list[vertex_trial][i]] = phi_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        mean_curvature_vertex[vertex_trial] = mean_curvature_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            mean_curvature_vertex[point_neighbor_list[vertex_trial][i]] = mean_curvature_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        sigma_vertex[vertex_trial] = sigma_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            sigma_vertex[point_neighbor_list[vertex_trial][i]] = sigma_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        for(int i=0; i<point_triangle_max[vertex_trial]; i++) {
            int j = point_triangle_list[vertex_trial][i];
            area_faces[j] = area_faces_original[j];
        }
    }
    steps_tested_displace_thread[thread_id][0] += 1;
	// cout << "Displace step end" << endl;
}

void MCMoves::TetherCut(int vertex_trial, int thread_id) {
    // Choose link at random, destroy it, and create new link joining other ends of triangle
    // Have to update entries of triangle_list, point_neighbor_list, point_neighbor_triangle, point_triangle_list, and point_neighbor_max for this
    // Select random vertex
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
	// Reject move if acting on rollers
    // int vertex_trial = 1000;
    // Select random link from avaliable
    int link_trial = local_generator.rand_select(point_neighbor_max[vertex_trial]-1);
    int vertex_trial_opposite = point_neighbor_list[vertex_trial][link_trial];

	/*
    cout << "Begin tether" << endl;
    cout << "Vertex trial: " << vertex_trial << endl;
    cout << "Link trial: " << link_trial << endl;
    cout << "Vertex opposite: " << vertex_trial_opposite << endl;
	*/

    int triangle_trial[2]; 
    int point_trial[2]; 
    int point_trial_position[2];

    // Find the triangles to be changed using point_neighbor_list
    triangle_trial[0] = point_neighbor_triangle[vertex_trial][link_trial][0];
    triangle_trial[1] = point_neighbor_triangle[vertex_trial][link_trial][1];

	/*
    cout << "Triangle " << triangle_trial[0] << ": " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
    cout << "Triangle " << triangle_trial[1] << ": " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
	*/
    // Check to see if vertex, vertex opposite are even on the two triangles
	/*
    int check_vertex = 0;
    int check_vertex_op = 0;
    for(int i=0; i<3; i++) {
        if(triangle_list[triangle_trial[0]][i] == vertex_trial) {
            check_vertex++;
        }
        if(triangle_list[triangle_trial[0]][i] == vertex_trial_opposite) {
            check_vertex_op++;
        }
        if(triangle_list[triangle_trial[1]][i] == vertex_trial) {
            check_vertex++;
        }
        if(triangle_list[triangle_trial[1]][i] == vertex_trial_opposite) {
            check_vertex_op++;
        }
    }

    if((check_vertex != 2) || (check_vertex_op != 2)) {
        cout << "Broken triangles" << endl;
        cout << "End tether" << endl;
        return;
    }
	*/
    // Find the two other points in the triangles using faces
    for(int i=0; i<2; i++) {
        if ((vertex_trial == triangle_list[triangle_trial[i]][0]) || (vertex_trial_opposite == triangle_list[triangle_trial[i]][0])) {
            if ((vertex_trial == triangle_list[triangle_trial[i]][1]) || (vertex_trial_opposite == triangle_list[triangle_trial[i]][1])) {
                point_trial[i] = triangle_list[triangle_trial[i]][2];
                point_trial_position[i] = 2;
            }
            else {
                point_trial[i] = triangle_list[triangle_trial[i]][1];
                point_trial_position[i] = 1;
            }
        }
        else {
            point_trial[i] = triangle_list[triangle_trial[i]][0];
            point_trial_position[i] = 0;
        }
    }
	int triangle_break[2];
    if((vertex_trial < 0) || (vertex_trial_opposite < 0) || (point_trial[0] < 0) || (point_trial[1] < 0) || (vertex_trial_opposite > vertices) || (point_trial[0] > vertices) || (point_trial[1] > vertices)) {
        /*
        cout << "It's breaking " << vertex_trial << " " << vertex_trial_opposite << " " << point_trial[0] << " " << point_trial[1] << endl;
        cout << link_triangle_list[point_trial[0]][point_trial[1]][0] << " " << link_triangle_list[point_trial[0]][point_trial[1]][1] << " " << link_triangle_list[point_trial[1]][point_trial[0]][0] << " " << link_triangle_list[point_trial[1]][point_trial[0]][1] << endl;
		cout << "Triangle " << triangle_trial[0] << " : " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
		cout << "Triangle " << triangle_trial[1] << " : " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
		cout << "Triangles that are breaking" << endl;
		if ((link_triangle_list[point_trial[0]][point_trial[1]][0] > 0) || (link_triangle_list[point_trial[0]][point_trial[1]][1] > 0)) {
			triangle_break[0] = link_triangle_list[point_trial[0]][point_trial[1]][0];
			triangle_break[1] = link_triangle_list[point_trial[0]][point_trial[1]][1];
			cout << "Triangle " << triangle_break[0] << " : " << triangle_list[triangle_break[0]][0] << " " << triangle_list[triangle_break[0]][1] << " " << triangle_list[triangle_break[0]][2] << endl;
			cout << "Triangle " << triangle_break[1] << " : " << triangle_list[triangle_break[1]][0] << " " << triangle_list[triangle_break[1]][1] << " " << triangle_list[triangle_break[1]][2] << endl;
		}
		if ((link_triangle_list[point_trial[1]][point_trial[1]][0] > 0) || (link_triangle_list[point_trial[1]][point_trial[0]][1] > 0)) {
			triangle_break[0] = link_triangle_list[point_trial[1]][point_trial[0]][0];
			triangle_break[1] = link_triangle_list[point_trial[1]][point_trial[0]][1];
			cout << "Triangle " << triangle_break[0] << " : " << triangle_list[triangle_break[0]][0] << " " << triangle_list[triangle_break[0]][1] << " " << triangle_list[triangle_break[0]][2] << endl;
			cout << "Triangle " << triangle_break[1] << " : " << triangle_list[triangle_break[1]][0] << " " << triangle_list[triangle_break[1]][1] << " " << triangle_list[triangle_break[1]][2] << endl;
		}
		cout << "Link trial " << link_trial << " neighbor max " << point_neighbor_max[vertex_trial] << endl;
		dumpLAMMPSConfig("lconfig_break");
        cout << "End tether" << endl;
        */
		steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if point_trial[0] and point_trial[1] are already linked
    for(int i=0; i<point_neighbor_max[point_trial[0]]; i++) {
        if(point_neighbor_list[point_trial[0]][i] == point_trial[1]) {
            steps_rejected_tether_thread[thread_id][0] += 1;
            steps_tested_tether_thread[thread_id][0] += 1;
            return;
        }
    }
    
    // cout << "Other points are: " << point_trial[0] << " " << point_trial[1] << " at " << point_trial_position[0] << " " << point_trial_position[1] << endl;
	// Check to see if the limits for maximum or minimum number of neighbors is exceeded
	if(((point_neighbor_max[vertex_trial]-1) == neighbor_min) || ((point_neighbor_max[vertex_trial_opposite]-1) == neighbor_min) || ((point_neighbor_max[point_trial[0]]+1) == neighbor_max) || ((point_neighbor_max[point_trial[1]]+1) == neighbor_max)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        // cout << "End tether" << endl;
		return;
	}
    
    // Check to see if any of the points are outside of the checkerboard set that vertex_trial is in
    if((checkerboard_index[vertex_trial] != checkerboard_index[vertex_trial_opposite]) || (checkerboard_index[vertex_trial] != checkerboard_index[point_trial[0]]) || (checkerboard_index[vertex_trial] != checkerboard_index[point_trial[1]])) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        // cout << "End tether" << endl;
		return;
    }

    // Check to see if trial points are too far apart
    double distance_point_trial = lengthLink(point_trial[0], point_trial[1]); 
    if ((distance_point_trial > 1.673) || (distance_point_trial < 1.00)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        // cout << "End tether" << endl;
        return;    
    }

    // Calculate detailed balance factor before everything changes
    // Basically, we now acc(v -> w) = gen(w->v) P(w)/gen(v->w) P(v)
    // Probability of gen(v->w) is 1/N*(1/Neighbors at vertex trial + 1/Neighbors at vertex_trial_opposite)
    // Similar for gen(w->v) except with point trial
    double db_factor = (1.0/(double(point_neighbor_max[point_trial[0]])+1.0)+1.0/(double(point_neighbor_max[point_trial[1]])+1.0))/(1.0/double(point_neighbor_max[vertex_trial])+1.0/double(point_neighbor_max[vertex_trial_opposite]));

    // Have all points needed, now just time to change triangle_list, point_neighbor_list, point_neighbor_triangle, point_triangle_list, point_neighbor_max, and point_triangle_max
    // Change triangle_list
    // Check orientation to see if consistent with before
    for(int i=0; i<3; i++) {
        if(triangle_list[triangle_trial[0]][i] == vertex_trial_opposite) {
            triangle_list[triangle_trial[0]][i] = point_trial[1];
            break;
        }
    }
    for(int i=0; i<3; i++) {
        if(triangle_list[triangle_trial[1]][i] == vertex_trial) {
            triangle_list[triangle_trial[1]][i] = point_trial[0];
            break;
        }
    }

	/*
    cout << "Triangle " << triangle_trial[0] << ": " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
    cout << "Triangle " << triangle_trial[1] << ": " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
	*/
    // Change point_neighbor_list and point_neighbor_triangle
    // Delete points
    int placeholder_nl = 0;
    int placeholder_neighbor1 = 0;
    int placeholder_neighbor2 = 0;
    // cout << "Neighbor list for point " << vertex_trial << " :";
    while(placeholder_nl < point_neighbor_max[vertex_trial]) {
        if(point_neighbor_list[vertex_trial][placeholder_nl] == vertex_trial_opposite) {
            // cout << " " << point_neighbor_list[vertex_trial][placeholder_nl] << " ";
            point_neighbor_list[vertex_trial][placeholder_nl] = point_neighbor_list[vertex_trial][point_neighbor_max[vertex_trial]-1];
            point_neighbor_triangle[vertex_trial][placeholder_nl][0] = point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]-1][0];
            point_neighbor_triangle[vertex_trial][placeholder_nl][1] = point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]-1][1];
            placeholder_neighbor1 = placeholder_nl;
            point_neighbor_list[vertex_trial][point_neighbor_max[vertex_trial]-1] = -1;
            // cout << " now " << point_neighbor_list[vertex_trial][placeholder_nl];
            placeholder_nl = neighbor_max;
            // cout << " SKIP ";
        }
        // cout << " " << point_neighbor_list[vertex_trial][placeholder_nl] << " ";
        placeholder_nl += 1;
    }
    // cout << " end " << endl;
    point_neighbor_max[vertex_trial] -= 1;
    placeholder_nl = 0;
    while(placeholder_nl < point_neighbor_max[vertex_trial_opposite]) {
        if(point_neighbor_list[vertex_trial_opposite][placeholder_nl] == vertex_trial) {
            point_neighbor_list[vertex_trial_opposite][placeholder_nl] = point_neighbor_list[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1];
            point_neighbor_triangle[vertex_trial_opposite][placeholder_nl][0] = point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1][0];
            point_neighbor_triangle[vertex_trial_opposite][placeholder_nl][1] = point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1][1];
            placeholder_neighbor2 = placeholder_nl;
            point_neighbor_list[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1] = -1;
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    point_neighbor_max[vertex_trial_opposite] -= 1;
    // Add points
    point_neighbor_list[point_trial[0]][point_neighbor_max[point_trial[0]]] = point_trial[1];
    point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]][0] = triangle_trial[0];
    point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]][1] = triangle_trial[1];
    point_neighbor_list[point_trial[1]][point_neighbor_max[point_trial[1]]] = point_trial[0];
    point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]][0] = triangle_trial[0];
    point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]][1] = triangle_trial[1];
    point_neighbor_max[point_trial[0]] += 1;
    point_neighbor_max[point_trial[1]] += 1;

    // Note that the definition of triangle_trial[0] and triangle_trial[1] have changed
    // Need to modify point_neighbor_triangle entries between vertex_trial and point_trial[1]
    // and vertex_trial_opposite and point_trial[0] to swap triangle_trial[1] to triangle_trial[0]
    // and triangle_trial[0] to triangle_trial[1] respectively
    // Placeholder values so places needed are saved
    int placeholder_remake[4] = {0,0,0,0};
    int placeholder_remake_01[4] = {0,0,0,0};
    // vertex_trial and point_trial[1]
    for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
        if(point_neighbor_list[vertex_trial][i] == point_trial[1]) {
            placeholder_remake[0] = i;
            if(point_neighbor_triangle[vertex_trial][i][0] == triangle_trial[1]) {
                point_neighbor_triangle[vertex_trial][i][0] = triangle_trial[0];
                placeholder_remake_01[0] = 0;
            }
            else if(point_neighbor_triangle[vertex_trial][i][1] == triangle_trial[1]) {
                point_neighbor_triangle[vertex_trial][i][1] = triangle_trial[0];
                placeholder_remake_01[0] = 1;
            }
        }
    }
    for(int i=0; i<point_neighbor_max[point_trial[1]]; i++) {
        if(point_neighbor_list[point_trial[1]][i] == vertex_trial) {
            placeholder_remake[1] = i;
            if(point_neighbor_triangle[point_trial[1]][i][0] == triangle_trial[1]) {
                point_neighbor_triangle[point_trial[1]][i][0] = triangle_trial[0];
                placeholder_remake_01[1] = 0;
            }
            else if(point_neighbor_triangle[point_trial[1]][i][1] == triangle_trial[1]) {
                point_neighbor_triangle[point_trial[1]][i][1] = triangle_trial[0];
                placeholder_remake_01[1] = 1;
            }
        }
    }
    // vertex_trial_opposite and point_trial[0]
    for(int i=0; i<point_neighbor_max[vertex_trial_opposite]; i++) {
        if(point_neighbor_list[vertex_trial_opposite][i] == point_trial[0]) {
            placeholder_remake[2] = i;
            if(point_neighbor_triangle[vertex_trial_opposite][i][0] == triangle_trial[0]) {
                point_neighbor_triangle[vertex_trial_opposite][i][0] = triangle_trial[1];
                placeholder_remake_01[2] = 0;
            }
            else if(point_neighbor_triangle[vertex_trial_opposite][i][1] == triangle_trial[0]) {
                point_neighbor_triangle[vertex_trial_opposite][i][1] = triangle_trial[1];
                placeholder_remake_01[2] = 1;
            }
        }
    }
    for(int i=0; i<point_neighbor_max[point_trial[0]]; i++) {
        if(point_neighbor_list[point_trial[0]][i] == vertex_trial_opposite) {
            placeholder_remake[3] = i;
            if(point_neighbor_triangle[point_trial[0]][i][0] == triangle_trial[0]) {
                point_neighbor_triangle[point_trial[0]][i][0] = triangle_trial[1];
                placeholder_remake_01[3] = 0;
            }
            else if(point_neighbor_triangle[point_trial[0]][i][1] == triangle_trial[0]) {
                point_neighbor_triangle[point_trial[0]][i][1] = triangle_trial[1];
                placeholder_remake_01[3] = 1;
            }
        }
    }

    // Change point_triangle_list
    placeholder_nl = 0;
    int placeholder_triangle1 = 0;
    int placeholder_triangle2 = 0;
    while(placeholder_nl < point_triangle_max[vertex_trial]) {
        if(point_triangle_list[vertex_trial][placeholder_nl] == triangle_trial[1]) {
            point_triangle_list[vertex_trial][placeholder_nl] = point_triangle_list[vertex_trial][point_triangle_max[vertex_trial]-1];
            placeholder_triangle1 = placeholder_nl;
            point_triangle_list[vertex_trial][point_triangle_max[vertex_trial]-1] = -1;
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    point_triangle_max[vertex_trial] -= 1;
    placeholder_nl = 0;
    while(placeholder_nl < point_triangle_max[vertex_trial_opposite]) {
        if(point_triangle_list[vertex_trial_opposite][placeholder_nl] == triangle_trial[0]) {
            point_triangle_list[vertex_trial_opposite][placeholder_nl] = point_triangle_list[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]-1];
            placeholder_triangle2 = placeholder_nl;
            point_triangle_list[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]-1] = -1;
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    point_triangle_max[vertex_trial_opposite] -= 1;
    // Add points
    point_triangle_list[point_trial[0]][point_triangle_max[point_trial[0]]] = triangle_trial[1];
    point_triangle_list[point_trial[1]][point_triangle_max[point_trial[1]]] = triangle_trial[0];
    point_triangle_max[point_trial[0]] += 1;
    point_triangle_max[point_trial[1]] += 1;

    // Evaluate energy difference
    // Evaluated at four nodes from two triangles changed
    double Phi_diff = 0; 
    double Phi_diff_bending = 0;
    double Phi_diff_phi = 0;
    // Energy due to surface area
    areaNode(triangle_trial[0]);
    areaNode(triangle_trial[1]);
    Phi_diff += gamma_surf[0]*(area_faces[triangle_trial[0]] - area_faces_original[triangle_trial[0]]);
    // cout << "Phi_diff after " << vertex_trial << " " << Phi_diff << endl;
    Phi_diff += gamma_surf[0]*(area_faces[triangle_trial[1]] - area_faces_original[triangle_trial[1]]);
    // cout << "Phi_diff after " << vertex_trial << " " << Phi_diff << endl;
    // Energy due to mean curvature
    energyNode(vertex_trial);
    energyNode(vertex_trial_opposite);
    energyNode(point_trial[0]);
    energyNode(point_trial[1]);
    Phi_diff_bending += phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    // cout << "Phi_diff_bending after " << vertex_trial << " " << Phi_diff_bending << endl;
    Phi_diff_bending += phi_vertex[vertex_trial_opposite] - phi_vertex_original[vertex_trial_opposite];
    // cout << "Phi_diff_bending after " << vertex_trial_opposite << " " << Phi_diff_bending << endl;
    Phi_diff_bending += phi_vertex[point_trial[0]] - phi_vertex_original[point_trial[0]];
    // cout << "Phi_diff_bending after " << point_trial[0] << " " << Phi_diff_bending << endl;
    Phi_diff_bending += phi_vertex[point_trial[1]] - phi_vertex_original[point_trial[1]];
    // cout << "Phi_diff_bending after " << point_trial[1] << " " << Phi_diff_bending << endl;
    Phi_diff += Phi_diff_bending;

	Phi_diff_phi += J_coupling[Ising_Array[vertex_trial]][Ising_Array[vertex_trial_opposite]]*ising_values[Ising_Array[vertex_trial]]*ising_values[Ising_Array[vertex_trial_opposite]];
	Phi_diff_phi -= J_coupling[Ising_Array[point_trial[0]]][Ising_Array[point_trial[1]]]*ising_values[Ising_Array[point_trial[0]]]*ising_values[Ising_Array[point_trial[1]]];

    Phi_diff += Phi_diff_phi;
    // cout << "Phi_diff is " << Phi_diff << endl;
    double Chance = local_generator.d();
    double Chance_factor = -T*log(Chance/db_factor);
    bool accept = false;

	/*
	// Debugging
    double Phi_ = Phi + Phi_diff;
	cout << "Before init Phi " << " " << Phi << endl;
	initializeEnergy();
    if (Phi > pow(10,9)) {
		Phi -= 2*pow(10,9);
	}
	cout.precision(10);
    cout << "Phi " << Phi << " and Phi_ " << Phi_ << endl;
	cout << "Points " << vertex_trial << " " << vertex_trial_opposite << " " << point_trial[0] << " " << point_trial[1] << endl;
	cout << "Phi before chance in tether cut " << std::scientific << (Phi - Phi_) << endl;
	Phi = Phi_ - Phi_diff;
	*/

    if((accept == true) || ((Chance_factor>Phi_diff) && (Phi_diff < pow(10,10)))) {
        // Accept move
        // New way that uses energy loop
        // Accept all trial values with for loop
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;

        // Update original values
        // Have all points needed, now just time to change triangle_list, point_neighbor_list, link_triangle_list, point_triangle_list, point_neighbor_max, and point_triangle_max
        // Change triangle_list
        triangle_list_original[triangle_trial[0]][0] = triangle_list[triangle_trial[0]][0];
        triangle_list_original[triangle_trial[0]][1] = triangle_list[triangle_trial[0]][1];
        triangle_list_original[triangle_trial[0]][2] = triangle_list[triangle_trial[0]][2];

        triangle_list_original[triangle_trial[1]][0] = triangle_list[triangle_trial[1]][0];
        triangle_list_original[triangle_trial[1]][1] = triangle_list[triangle_trial[1]][1];
        triangle_list_original[triangle_trial[1]][2] = triangle_list[triangle_trial[1]][2];

		/*
        cout << "End accept " << vertex_trial << " " << vertex_trial_opposite << " " << point_trial[0] << " " << point_trial[1] << endl;
        cout << link_triangle_list_original[vertex_trial][vertex_trial_opposite][0] << " " << link_triangle_list_original[vertex_trial][vertex_trial_opposite][1] << " " << link_triangle_list_original[vertex_trial_opposite][vertex_trial][0] << " " << link_triangle_list_original[vertex_trial_opposite][vertex_trial][1] << endl;
        cout << link_triangle_list_original[point_trial[0]][point_trial[1]][0] << " " << link_triangle_list_original[point_trial[0]][point_trial[1]][1] << " " << link_triangle_list_original[point_trial[1]][point_trial[0]][0] << " " << link_triangle_list_original[point_trial[1]][point_trial[0]][1] << endl;
		cout << "Triangle " << triangle_trial[0] << " : " << triangle_list_original[triangle_trial[0]][0] << " " << triangle_list_original[triangle_trial[0]][1] << " " << triangle_list_original[triangle_trial[0]][2] << endl;
		cout << "Triangle " << triangle_trial[1] << " : " << triangle_list_original[triangle_trial[1]][0] << " " << triangle_list_original[triangle_trial[1]][1] << " " << triangle_list_original[triangle_trial[1]][2] << endl;
		*/
        // Check to see if point_trial[0], point_trial[1] are even on the two triangles
		/*
        int check_0 = 0;
        int check_1 = 0;
        for(int i=0; i<3; i++) {
            if(triangle_list_original[triangle_trial[0]][i] == point_trial[0]) {
                check_0++;
            }
            if(triangle_list_original[triangle_trial[0]][i] == point_trial[1]) {
                check_1++;
            }
            if(triangle_list_original[triangle_trial[1]][i] == point_trial[0]) {
                check_0++;
            }
            if(triangle_list_original[triangle_trial[1]][i] == point_trial[1]) {
                check_1++;
            }
        }

        if((check_0 != 2) || (check_1 != 2)) {
            cout << "Broken triangles in accept" << endl;
        }
		*/
        // Change point_neighbor_list
        // Delete points
        point_neighbor_list_original[vertex_trial][placeholder_neighbor1] = point_neighbor_list_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1];
        point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][0] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1][0];
        point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][1] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1][1];
        point_neighbor_list_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1] = -1;
        point_neighbor_max_original[vertex_trial] -= 1;
        point_neighbor_list_original[vertex_trial_opposite][placeholder_neighbor2] = point_neighbor_list_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1];
        point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][0] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1][0];
        point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][1] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1][1];
        point_neighbor_list_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1] = -1;
        point_neighbor_max_original[vertex_trial_opposite] -= 1;

        // Add points
        point_neighbor_list_original[point_trial[0]][point_neighbor_max_original[point_trial[0]]] = point_trial[1];
        point_neighbor_list_original[point_trial[1]][point_neighbor_max_original[point_trial[1]]] = point_trial[0];
        point_neighbor_triangle_original[point_trial[0]][point_neighbor_max_original[point_trial[0]]][0] = triangle_trial[0];
        point_neighbor_triangle_original[point_trial[0]][point_neighbor_max_original[point_trial[0]]][1] = triangle_trial[1];
        point_neighbor_triangle_original[point_trial[1]][point_neighbor_max_original[point_trial[1]]][0] = triangle_trial[0];
        point_neighbor_triangle_original[point_trial[1]][point_neighbor_max_original[point_trial[1]]][1] = triangle_trial[1];
        point_neighbor_max_original[point_trial[0]] += 1;
        point_neighbor_max_original[point_trial[1]] += 1;

        // Points changed due to redefinitions in triangle_trial[0] and triangle_trial[1]
        point_neighbor_triangle_original[vertex_trial][placeholder_remake[0]][placeholder_remake_01[0]] = triangle_trial[0];
        point_neighbor_triangle_original[point_trial[1]][placeholder_remake[1]][placeholder_remake_01[1]] = triangle_trial[0];
        point_neighbor_triangle_original[vertex_trial_opposite][placeholder_remake[2]][placeholder_remake_01[2]] = triangle_trial[1];
        point_neighbor_triangle_original[point_trial[0]][placeholder_remake[3]][placeholder_remake_01[3]] = triangle_trial[1];

        // Change point_triangle_list
        point_triangle_list_original[vertex_trial][placeholder_triangle1] = point_triangle_list_original[vertex_trial][point_triangle_max_original[vertex_trial]-1];
        point_triangle_list_original[vertex_trial][point_triangle_max_original[vertex_trial]-1] = -1;
        point_triangle_max_original[vertex_trial] -= 1;
        point_triangle_list_original[vertex_trial_opposite][placeholder_triangle2] = point_triangle_list_original[vertex_trial_opposite][point_triangle_max_original[vertex_trial_opposite]-1];
        point_triangle_list_original[vertex_trial_opposite][point_triangle_max_original[vertex_trial_opposite]-1] = -1;
        point_triangle_max_original[vertex_trial_opposite] -= 1;
        // Add points
        point_triangle_list_original[point_trial[0]][point_triangle_max_original[point_trial[0]]] = triangle_trial[1];
        point_triangle_list_original[point_trial[1]][point_triangle_max_original[point_trial[1]]] = triangle_trial[0];
        point_triangle_max_original[point_trial[0]] += 1;
        point_triangle_max_original[point_trial[1]] += 1;

        // Update energy values
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_opposite] = phi_vertex[vertex_trial_opposite];
        phi_vertex_original[point_trial[0]] = phi_vertex[point_trial[0]];
        phi_vertex_original[point_trial[1]] = phi_vertex[point_trial[1]];
        mean_curvature_vertex_original[vertex_trial] = mean_curvature_vertex[vertex_trial];
        mean_curvature_vertex_original[vertex_trial_opposite] = mean_curvature_vertex[vertex_trial_opposite];
        mean_curvature_vertex_original[point_trial[0]] = mean_curvature_vertex[point_trial[0]];
        mean_curvature_vertex_original[point_trial[1]] = mean_curvature_vertex[point_trial[1]];
        sigma_vertex_original[vertex_trial] = sigma_vertex[vertex_trial];
        sigma_vertex_original[vertex_trial_opposite] = sigma_vertex[vertex_trial_opposite];
        sigma_vertex_original[point_trial[0]] = sigma_vertex[point_trial[0]];
        sigma_vertex_original[point_trial[1]] = sigma_vertex[point_trial[1]];

        double Area_total_diff = 0.0;
        Area_total_diff += area_faces[triangle_trial[0]] - area_faces_original[triangle_trial[0]];
        Area_total_diff += area_faces[triangle_trial[1]] - area_faces_original[triangle_trial[1]];
        Area_diff_thread[thread_id][0] += Area_total_diff;

        area_faces_original[triangle_trial[0]] = area_faces[triangle_trial[0]];
        area_faces_original[triangle_trial[1]] = area_faces[triangle_trial[1]]; 
            
    }
    else {
        // cout << "Reject move at " << vertex_trial << endl;
        steps_rejected_tether_thread[thread_id][0] += 1;
        // Change new values to original values
        // Have all points needed, now just time to change triangle_list, point_neighbor_list, link_triangle_list, point_triangle_list, point_neighbor_max, and point_triangle_max
        // Change triangle_list
        triangle_list[triangle_trial[0]][0] = triangle_list_original[triangle_trial[0]][0];
        triangle_list[triangle_trial[0]][1] = triangle_list_original[triangle_trial[0]][1];
        triangle_list[triangle_trial[0]][2] = triangle_list_original[triangle_trial[0]][2];

        triangle_list[triangle_trial[1]][0] = triangle_list_original[triangle_trial[1]][0];
        triangle_list[triangle_trial[1]][1] = triangle_list_original[triangle_trial[1]][1];
        triangle_list[triangle_trial[1]][2] = triangle_list_original[triangle_trial[1]][2];

        // Change point_neighbor_list
        // Add points
        point_neighbor_list[vertex_trial][placeholder_neighbor1] = point_neighbor_list_original[vertex_trial][placeholder_neighbor1];
        point_neighbor_triangle[vertex_trial][placeholder_neighbor1][0] = point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][0];
        point_neighbor_triangle[vertex_trial][placeholder_neighbor1][1] = point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][1];        
        point_neighbor_list[vertex_trial][point_neighbor_max[vertex_trial]] = point_neighbor_list_original[vertex_trial][point_neighbor_max[vertex_trial]];
        point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]][0] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max[vertex_trial]][0];
        point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]][1] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max[vertex_trial]][1];
        point_neighbor_max[vertex_trial] += 1;
        point_neighbor_list[vertex_trial_opposite][placeholder_neighbor2] = point_neighbor_list_original[vertex_trial_opposite][placeholder_neighbor2];
        point_neighbor_triangle[vertex_trial_opposite][placeholder_neighbor2][0] = point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][0];
        point_neighbor_triangle[vertex_trial_opposite][placeholder_neighbor2][1] = point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][1];        
        point_neighbor_list[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]] = point_neighbor_list_original[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]];
        point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][0] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][0];
        point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][1] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][1];
        point_neighbor_max[vertex_trial_opposite] += 1;

        // Delete points
        point_neighbor_list[point_trial[0]][point_neighbor_max_original[point_trial[0]]] = -1;
        point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]-1][0] = -1;
        point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]-1][1] = -1;
        point_neighbor_list[point_trial[1]][point_neighbor_max_original[point_trial[1]]] = -1;
        point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]-1][0] = -1;
        point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]-1][1] = -1;
        point_neighbor_max[point_trial[0]] -= 1;
        point_neighbor_max[point_trial[1]] -= 1;

        // Points changed due to redefinitions in triangle_trial[0] and triangle_trial[1]
        point_neighbor_triangle[vertex_trial][placeholder_remake[0]][placeholder_remake_01[0]] = point_neighbor_triangle_original[vertex_trial][placeholder_remake[0]][placeholder_remake_01[0]];
        point_neighbor_triangle[point_trial[1]][placeholder_remake[1]][placeholder_remake_01[1]] = point_neighbor_triangle_original[point_trial[1]][placeholder_remake[1]][placeholder_remake_01[1]];
        point_neighbor_triangle[vertex_trial_opposite][placeholder_remake[2]][placeholder_remake_01[2]] = point_neighbor_triangle_original[vertex_trial_opposite][placeholder_remake[2]][placeholder_remake_01[2]];
        point_neighbor_triangle[point_trial[0]][placeholder_remake[3]][placeholder_remake_01[3]] = point_neighbor_triangle_original[point_trial[0]][placeholder_remake[3]][placeholder_remake_01[3]];

        // Change point_triangle_list
        point_triangle_list[vertex_trial][placeholder_triangle1] = point_triangle_list_original[vertex_trial][placeholder_triangle1];
        point_triangle_list[vertex_trial][point_triangle_max[vertex_trial]] = point_triangle_list_original[vertex_trial][point_triangle_max[vertex_trial]];
        point_triangle_max[vertex_trial] += 1;
        point_triangle_list[vertex_trial_opposite][placeholder_triangle2] = point_triangle_list_original[vertex_trial_opposite][placeholder_triangle2];
        point_triangle_list[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]] = point_triangle_list_original[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]];
        point_triangle_max[vertex_trial_opposite] += 1;
        // Delete points
        point_triangle_list[point_trial[0]][point_triangle_max_original[point_trial[0]]] = -1;
        point_triangle_list[point_trial[1]][point_triangle_max_original[point_trial[1]]] = -1;
        point_triangle_max[point_trial[0]] -= 1;
        point_triangle_max[point_trial[1]] -= 1;

        // Update energy values
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_opposite] = phi_vertex_original[vertex_trial_opposite];
        phi_vertex[point_trial[0]] = phi_vertex_original[point_trial[0]];
        phi_vertex[point_trial[1]] = phi_vertex_original[point_trial[1]];
        mean_curvature_vertex[vertex_trial] = mean_curvature_vertex_original[vertex_trial];
        mean_curvature_vertex[vertex_trial_opposite] = mean_curvature_vertex_original[vertex_trial_opposite];
        mean_curvature_vertex[point_trial[0]] = mean_curvature_vertex_original[point_trial[0]];
        mean_curvature_vertex[point_trial[1]] = mean_curvature_vertex_original[point_trial[1]];
        sigma_vertex[vertex_trial] = sigma_vertex_original[vertex_trial];
        sigma_vertex[vertex_trial_opposite] = sigma_vertex_original[vertex_trial_opposite];
        sigma_vertex[point_trial[0]] = sigma_vertex_original[point_trial[0]];
        sigma_vertex[point_trial[1]] = sigma_vertex_original[point_trial[1]];
   
        area_faces[triangle_trial[0]] = area_faces_original[triangle_trial[0]];
        area_faces[triangle_trial[1]] = area_faces_original[triangle_trial[1]]; 

        
    }
    steps_tested_tether_thread[thread_id][0] += 1;
    // cout << "End tether" << endl;

	/*
    Phi_ = Phi;
	initializeEnergy();
	cout.precision(17);
	cout << "Phi after chance in tether cut" << std::scientific << (Phi - Phi_) << endl;
	*/
    /* 
    cout << "At end of tether cut!" << endl;
    int loop[4] = {vertex_trial, vertex_trial_opposite, point_trial[0], point_trial[1]};
    cout << "Triangle 1: " << triangle_list_original[triangle_trial[0]][0] << " " << triangle_list_original[triangle_trial[0]][1] << " " << triangle_list_original[triangle_trial[0]][2] << endl;
    cout << "Triangle 2: " << triangle_list_original[triangle_trial[1]][0] << " " << triangle_list_original[triangle_trial[1]][1] << " " << triangle_list_original[triangle_trial[1]][2] << endl;
    cout << "Triangle 1: " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
    cout << "Triangle 2: " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
    for(int j_loop=0; j_loop<4; j_loop++) {
        int i = loop[j_loop];
        cout << "Max neighbors at " << i << " is " << point_neighbor_max[i] << endl;
        cout << "Max triangles at " << i << " is " << point_triangle_max[i] << endl;
    }

    
    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j_loop=0; j_loop<4; j_loop++) { 
            int i = loop[i_loop];
            int j = loop[j_loop];
            cout << "Link list at " << i << " " << j << " is 1: " << link_triangle_list[i][j][0] << " and 2: " << link_triangle_list[i][j][1] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j=0; j<10; j++){
            int i = loop[i_loop];
            cout << "Neighbor list at " << i << " entry " << j << " :" << point_neighbor_list[i][j] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        int i = loop[i_loop];
        cout << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<10; j++) {
            cout << point_triangle_list[i][j] << " ";
        }
        cout << endl;
    }
    
    for(int j_loop=0; j_loop<4; j_loop++) {
        int i = loop[j_loop];
        cout << "Max neighbors at " << i << " is " << point_neighbor_max_original[i] << endl;
        cout << "Max triangles at " << i << " is " << point_triangle_max_original[i] << endl;
    }

    
    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j_loop=0; j_loop<4; j_loop++) { 
            int i = loop[i_loop];
            int j = loop[j_loop];
            cout << "Link list at " << i << " " << j << " is 1: " << link_triangle_list_original[i][j][0] << " and 2: " << link_triangle_list_original[i][j][1] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j=0; j<10; j++){
            int i = loop[i_loop];
            cout << "Neighbor list at " << i << " entry " << j << " :" << point_neighbor_list_original[i][j] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        int i = loop[i_loop];
        cout << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<10; j++) {
            cout << point_triangle_list_original[i][j] << " ";
        }
        cout << endl;
    }
    */
    
}

void MCMoves::ChangeMassNonCon(int vertex_trial, int thread_id) {
    // Pick random site and change the Ising array value
    // Non-mass conserving
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
    // Have to have implementation that chooses from within the same checkerboard set
    if(Ising_Array[vertex_trial] == 2) {
        // If protein node is selected, do a moveProtein_gen move instead
        moveProtein_gen(vertex_trial, thread_id);
        return;
	} 
    // Change spin
    int Ising_Array_trial = 0;
    if(Ising_Array[vertex_trial] == 0){
        Ising_Array_trial = 1;
    }

    double Phi_diff_phi = 0;

    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    double Phi_diff_mag = -h_external*(ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial]*sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    double Phi_diff = Phi_diff_phi+Phi_diff_bending+Phi_diff_mag;
    double Chance = local_generator.d();
    if(Chance<exp(-Phi_diff/T)){
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Mass_diff_thread[thread_id][0] += (Ising_Array_trial-Ising_Array[vertex_trial]);
        Magnet_diff_thread[thread_id][0] += (ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);
        Ising_Array[vertex_trial] = Ising_Array_trial;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;
}

void MCMoves::ChangeMassNonConGL(int vertex_trial) {
    // Pick random site and change the Ising array value per Glauber dynamics
    // Non-mass conserving
    Saru& local_generator = generators[omp_get_thread_num()];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
    while (Ising_Array[vertex_trial] == 2) {
		vertex_trial = local_generator.rand_select(vertices-1);
	}
    // Change spin
    int Ising_Array_trial = 0;
    if(Ising_Array[vertex_trial] == 0){
        Ising_Array_trial = 1;
    }

    double Phi_magnet = 0;

    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial];
        Phi_magnet -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    Phi_magnet -= h_external*(ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial]*sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double Phi_diff = Phi_magnet + phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    double Chance = local_generator.d();
    if(Chance<(1.0/(1.0+exp(Phi_diff/T)))) {
        #pragma omp atomic
        Phi += Phi_diff;
        #pragma omp atomic
        Mass += (Ising_Array_trial-Ising_Array[vertex_trial]);
        #pragma omp atomic
        Magnet += (ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);
        Ising_Array[vertex_trial] = Ising_Array_trial;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
    }
    
    else {
        #pragma omp atomic
        steps_rejected_mass += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];    
    }
    #pragma omp atomic
    steps_tested_mass += 1;
}

void MCMoves::ChangeMassCon(int vertex_trial, int thread_id) {
    // Pick random site and attemp to swap Ising array value with array in random nearest neighbor direction
    // Mass conserving
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
    if(Ising_Array[vertex_trial] == 2) {
        // If protein node is selected, do a moveProtein_gen move instead
        moveProtein_gen(vertex_trial, thread_id);
        return;
	} 
    // Pick random direction
    int link_trial = local_generator.rand_select(point_neighbor_max[vertex_trial]-1);
    int vertex_trial_opposite = point_neighbor_list[vertex_trial][link_trial];

    // For now reject if the neighboring sites have the same array value or protein type
    if(Ising_Array[vertex_trial] == Ising_Array[vertex_trial_opposite]) {
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
    int Ising_Array_trial_1 = Ising_Array[vertex_trial_opposite];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_diff_phi = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_opposite]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial_opposite]][Ising_Array[point_neighbor_list[vertex_trial_opposite][j]]]*ising_values[Ising_Array[vertex_trial_opposite]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[vertex_trial_opposite][j]]]*ising_values[Ising_Array_trial_2];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial_opposite][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[vertex_trial_opposite]-spon_curv[Ising_Array_trial_2];
    phi_vertex[vertex_trial_opposite] = k_b[Ising_Array_trial_2]*sigma_vertex[vertex_trial_opposite]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    Phi_diff_phi += (J_coupling[Ising_Array_trial_1][Ising_Array[vertex_trial_opposite]]*ising_values[Ising_Array_trial_1]-J_coupling[Ising_Array[vertex_trial]][Ising_Array[vertex_trial_opposite]]*ising_values[Ising_Array[vertex_trial]])*ising_values[Ising_Array[vertex_trial_opposite]];
    Phi_diff_phi += (J_coupling[Ising_Array_trial_2][Ising_Array[vertex_trial]]*ising_values[Ising_Array_trial_2]-J_coupling[Ising_Array[vertex_trial_opposite]][Ising_Array[vertex_trial]]*ising_values[Ising_Array[vertex_trial_opposite]])*ising_values[Ising_Array[vertex_trial]];

    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff_bending += phi_vertex[vertex_trial_opposite] - phi_vertex_original[vertex_trial_opposite];
    double Phi_diff = Phi_diff_bending+Phi_diff_phi;
    double Chance = local_generator.d();
    if(Chance<exp(-Phi_diff/T)){
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[vertex_trial_opposite] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_opposite] = phi_vertex[vertex_trial_opposite];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_opposite] = phi_vertex_original[vertex_trial_opposite];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;

}

void MCMoves::ChangeMassConNL() {
    // Pick random site and attemp to swap Ising array value with another nonlocal site
    // Mass conserving
    // Can't think of a good parallel implementation for now
    Saru& local_generator = generators[omp_get_thread_num()];
    int vertex_trial = local_generator.rand_select(vertices-1);
    // Pick random site with opposite spin
    int vertex_trial_2 = local_generator.rand_select(vertices-1);

    // Keep generating new trial values if Ising values are the same
    while((Ising_Array[vertex_trial] == Ising_Array[vertex_trial_2]) || (Ising_Array[vertex_trial] == 2) || (Ising_Array[vertex_trial_2] == 2)) {
    	vertex_trial = local_generator.rand_select(vertices-1);
        vertex_trial_2 = local_generator.rand_select(vertices-1);   		 
    }    

    // Set trial values
    int Ising_Array_trial_1 = Ising_Array[vertex_trial_2];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_magnet = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_magnet -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_2]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial_2]][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array[vertex_trial_2]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array_trial_2];
        Phi_magnet -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[vertex_trial_2]-spon_curv[Ising_Array_trial_2];
    phi_vertex[vertex_trial_2] = k_b[Ising_Array_trial_2]*sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    // Now check for self contribution
    // First check to see if sites are neighboring
    int check_double_count = link_triangle_test(vertex_trial, vertex_trial_2);
    if(check_double_count == 1) {
        Phi_magnet += (J_coupling[Ising_Array_trial_1][Ising_Array[vertex_trial_2]]*ising_values[Ising_Array_trial_1]-J_coupling[Ising_Array[vertex_trial]][Ising_Array[vertex_trial_2]]*ising_values[Ising_Array[vertex_trial]])*ising_values[Ising_Array[vertex_trial_2]];
        Phi_magnet += (J_coupling[Ising_Array_trial_2][Ising_Array[vertex_trial]]*ising_values[Ising_Array_trial_2]-J_coupling[Ising_Array[vertex_trial_2]][Ising_Array[vertex_trial]]*ising_values[Ising_Array[vertex_trial_2]])*ising_values[Ising_Array[vertex_trial]];
    }    

    double Phi_diff = Phi_magnet + phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff += phi_vertex[vertex_trial_2] - phi_vertex_original[vertex_trial_2];
    double Chance = local_generator.d();
    if(Chance<exp(-Phi_diff/T)){
        Phi += Phi_diff;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[vertex_trial_2] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_2] = phi_vertex[vertex_trial_2];
    }
    
    else {
        steps_rejected_mass += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_2] = phi_vertex_original[vertex_trial_2];    
    }
    steps_tested_mass += 1;

}

void MCMoves::MoveProteinGen(int vertex_trial, int thread_id) {
    // Pick random protein and attempt to move it in the y-direction
    // As protein's not merging, don't let them
    Saru& local_generator = generators[thread_id];
    // cout << "Protein trial " << protein_trial << " ";
    // Pick direction
    // Instead just go with one it's neighbors
    int direction_trial = local_generator.rand_select(point_neighbor_max[vertex_trial]-1);
    int center_trial = point_neighbor_list[vertex_trial][direction_trial]; 
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
    int Ising_Array_trial_1 = Ising_Array[center_trial];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_diff_phi = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[center_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[center_trial]][Ising_Array[point_neighbor_list[center_trial][j]]]*ising_values[Ising_Array[center_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[center_trial][j]]]*ising_values[Ising_Array_trial_2];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[center_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[center_trial]-spon_curv[Ising_Array_trial_2];
    phi_vertex[center_trial] = k_b[Ising_Array_trial_2]*sigma_vertex[center_trial]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    Phi_diff_phi += (J_coupling[Ising_Array_trial_1][Ising_Array[center_trial]]*ising_values[Ising_Array_trial_1]-J_coupling[Ising_Array[vertex_trial]][Ising_Array[center_trial]]*ising_values[Ising_Array[vertex_trial]])*ising_values[Ising_Array[center_trial]];
    Phi_diff_phi += (J_coupling[Ising_Array_trial_2][Ising_Array[vertex_trial]]*ising_values[Ising_Array_trial_2]-J_coupling[Ising_Array[center_trial]][Ising_Array[vertex_trial]]*ising_values[Ising_Array[center_trial]])*ising_values[Ising_Array[vertex_trial]];
    
    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff_bending += phi_vertex[center_trial] - phi_vertex_original[center_trial];
    double Chance = local_generator.d();
    double db_factor = double(point_neighbor_max[vertex_trial])/double(point_neighbor_max[center_trial]);
    double Chance_factor = -T*log(Chance/db_factor);
    double Phi_diff = Phi_diff_phi+Phi_diff_bending;

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(Chance_factor>Phi_diff) {
        accept = true;
    }

    // cout << Phi_magnet << endl;
    // cout << Chance << " " << exp(-Phi_magnet/T) << endl;
    if(accept == true) {
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[center_trial] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[center_trial] = phi_vertex[center_trial];
        protein_node[vertex_trial] = case_center;
        protein_node[center_trial] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[center_trial] = phi_vertex_original[center_trial];
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
    int Ising_Array_trial_1 = Ising_Array[vertex_trial_2];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_diff_phi = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_2]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial_2]][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array[vertex_trial_2]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array_trial_2];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[vertex_trial_2]-spon_curv[Ising_Array_trial_2];
    phi_vertex[vertex_trial_2] = k_b[Ising_Array_trial_2]*sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff_bending += phi_vertex[vertex_trial_2] - phi_vertex_original[vertex_trial_2];
    double Chance = local_generator.d();
    double Chance_factor = -T*log(Chance);
    double Phi_diff = Phi_diff_phi+Phi_diff_bending;

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(Chance_factor>Phi_diff) {
        accept = true;
    }

    // cout << Phi_magnet << endl;
    // cout << Chance << " " << exp(-Phi_magnet/T) << endl;
    if(accept == true) {
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[vertex_trial_2] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_2] = phi_vertex[vertex_trial_2];
        protein_node[vertex_trial] = case_center;
        protein_node[vertex_trial_2] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_2] = phi_vertex_original[vertex_trial_2];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void MCMoves::ChangeArea() {
// Attempt to change Length_x and Length_y uniformly
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
    Length_x = Length_x_base*scale_xy_trial;
    Length_y = Length_y_base*scale_xy_trial;
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
    double Phi_ = Phi;
    double Phi_bending_ = Phi_bending;
    double Phi_phi_ = Phi_phi;
    double Area_total_ = Area_total;
    // Recompute energy
    // Note that this version doesn't override all variables
    initializeEnergy_scale();
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[2] += time_span.count();
    // Now accept/reject
    t1_area = chrono::steady_clock::now();
    double Chance = generator.d();
    double Phi_diff = Phi-Phi_-T*2*vertices*log(scale_xy_trial/scale_xy);
    if((Chance<exp(-Phi_diff/T)) && (Phi_diff < pow(10,10))) {
        scale_xy = scale_xy_trial;
        Length_x_old = Length_x;
        Length_y_old = Length_y;
        box_x = Length_x/double(nl_x); 
        box_y = Length_y/double(nl_y); 
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            phi_vertex_original[i] = phi_vertex[i];
        }
        #pragma omp parallel for
        for(int i=0; i<faces; i++) {
            area_faces_original[i] = area_faces[i];
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            mean_curvature_vertex_original[i] = mean_curvature_vertex[i]; 
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            sigma_vertex_original[i] = sigma_vertex[i]; 
        }
    }
    else {
        steps_rejected_area++;
        Phi = Phi_;
        Phi_bending = Phi_bending_;
        Phi_phi = Phi_phi_;
        Area_total = Area_total_;
        Length_x = Length_x_old;
        Length_y = Length_y_old;
        generateNeighborList();
        box_x = Length_x/double(nl_x); 
        box_y = Length_y/double(nl_y); 
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            phi_vertex[i] = phi_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<faces; i++) {
            area_faces[i] = area_faces_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            mean_curvature_vertex[i] = mean_curvature_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            sigma_vertex[i] = sigma_vertex_original[i];
        }
    }
    steps_tested_area++;
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[3] += time_span.count();
}
