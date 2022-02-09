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
#include "neighborlist.hpp"
using namespace std;

NeighborList::NeighborList() {
    // Constructor
    // Does nothing
}

NeighborList::~NeighborList() {
    // Destructor
    // Does nothing
}

void NeighborList::GenerateNeighborList(MembraneMC& sys) {    
	// Generates neighbor list from current configuration
    // Check to see if nl_x, nl_y are different
    // If yes, then continue, if not don't need to rebuild
    int nl_x_trial = int(sys.lengths[0])-1;
    int nl_y_trial = int(sys.lengths[1])-1;
    if(((nl_x_trial >= nl_x) && (nl_y_trial >= nl_y)) && ((nl_x_trial <= (nl_x+4)) && (nl_y_trial <= (nl_y+4)))) {
        return;
    }
    // Clear current lists
    neighbor_list.clear();
    neighbor_list_index.clear();
    neighbors.clear();
	// Set up list
	vector<int> list;
    // Evaluate current values of nl_x, nl_y as that can change
    // Ideal box size is a little larger than 1.0. Get that by converting
    // sys.lengths[0], sys.lengths[1] to int then substracting by 2
    nl_x = int(sys.lengths[0])-2;
    nl_y = int(sys.lengths[1])-2;
    nl_z = int(sys.lengths[2]*2.0)-2;
    // New box size values
    box_x = sys.lengths[0]/double(nl_x); 
    box_y = sys.lengths[1]/double(nl_y); 
    box_z = 2*sys.lengths[2]/double(nl_z); 

    // Resize
    neighbor_list_index.resize(sys.vertices);
    neighbor_list.resize(nl_x*nl_y*nl_z);
    #pragma omp parallel for
	for(int i=0; i<nl_x*nl_y*nl_z; i++) {
    	neighbor_list[i] = list;
	}
	// Loop through membrane particles
    int index_particles[sys.vertices];
    int index_particles_add[sys.vertices];
    index_particles_max_nl.resize(nl_x*nl_y*nl_z,0);
    #pragma omp parallel for
	for(int i=0; i<sys.vertices; i++) {
		// Determine index
		int index_x = int(sys.lengths[0]*(sys.radii_tri[i][0]+0.5)/box_x);
		int index_y = int(sys.lengths[1]*(sys.radii_tri[i][1]+0.5)/box_y);
		int index_z = int((sys.radii_tri[i][2]+sys.lengths[2])/box_z);
        if(index_x == nl_x) {
            index_x -= 1;
        }
        if(index_y == nl_y) {
            index_y -= 1;
        }
        int particle_index = index_x + index_y*nl_x + index_z*nl_x*nl_y;
		index_particles[i] = particle_index;
        #pragma omp atomic capture 
        index_particles_add[i] = index_particles_max_nl[particle_index]++;
        // cout << index << endl;
	}

    #pragma omp parallel for
    for(int i=0; i<sys.vertices; i++) {
		neighbor_list_index[i] = index_particles[i];
    }
    // First resize neighbor_list in parallel
    #pragma omp parallel for
    for(int i=0; i<nl_x*nl_y*nl_z; i++) {
        neighbor_list[i].resize(index_particles_max_nl[i]);
	}
    // Now add particles to neighbor_list in parallel
    // As index of adding particles is now, done trivially
    #pragma omp parallel for
    for(int i=0; i<sys.vertices; i++) {
        neighbor_list[index_particles[i]][index_particles_add[i]] = i;
	}

	// Determine neighboring boxes
	vector<int> list_int;
    neighbors.resize(nl_x*nl_y*nl_z);
    #pragma omp parallel for
	for(int i=0; i<nl_x*nl_y*nl_z; i++) {
    	neighbors[i] = list_int;
	}
    #pragma omp parallel for
    for(int i=0; i<nl_x*nl_y*nl_z; i++) {
		int index_x = i % nl_x;
		int index_y = (i%(nl_x*nl_y))/nl_x;
		int index_z = i/(nl_x*nl_y);
		// Sweep over neighbors
		int index_z_down = index_z-1;
		int index_z_up = index_z+1;
		int index_y_down = ((index_y-1)%nl_y+nl_y)%nl_y;
		int index_y_up = ((index_y+1)%nl_y+nl_y)%nl_y;
		int index_x_down = ((index_x-1)%nl_x+nl_x)%nl_x;
		int index_x_up = ((index_x+1)%nl_x+nl_x)%nl_x;
		// Checking below, middle, and above cases
		// For each one, xy stencil check will be in this order
		// 1 2 3
		// 4 5 6
		// 7 8 9
		// Below
		if(index_z_down >= 0) {
			// 1
            neighbors[i].push_back(index_x_down + index_y_up*nl_x + index_z_down*nl_x*nl_y);
			// 2
            neighbors[i].push_back(index_x + index_y_up*nl_x + index_z_down*nl_x*nl_y);
			// 3
            neighbors[i].push_back(index_x_up + index_y_up*nl_x + index_z_down*nl_x*nl_y);
			// 4
            neighbors[i].push_back(index_x_down + index_y*nl_x + index_z_down*nl_x*nl_y);
			// 5
			neighbors[i].push_back(index_x + index_y*nl_x + index_z_down*nl_x*nl_y);
			// 6
            neighbors[i].push_back(index_x_up + index_y*nl_x + index_z_down*nl_x*nl_y);
			// 7
            neighbors[i].push_back(index_x_down + index_y_down*nl_x + index_z_down*nl_x*nl_y);
			// 8
            neighbors[i].push_back(index_x + index_y_down*nl_x + index_z_down*nl_x*nl_y);
			// 9
            neighbors[i].push_back(index_x_up + index_y_down*nl_x + index_z_down*nl_x*nl_y);
		}
		// Middle
		// 1
        neighbors[i].push_back(index_x_down + index_y_up*nl_x + index_z*nl_x*nl_y);
		// 2
        neighbors[i].push_back(index_x + index_y_up*nl_x + index_z*nl_x*nl_y);
		// 3
        neighbors[i].push_back(index_x_up + index_y_up*nl_x + index_z*nl_x*nl_y);
		// 4
        neighbors[i].push_back(index_x_down + index_y*nl_x + index_z*nl_x*nl_y);
		// 5
		neighbors[i].push_back(index_x + index_y*nl_x + index_z*nl_x*nl_y);
		// 6
        neighbors[i].push_back(index_x_up + index_y*nl_x + index_z*nl_x*nl_y);
		// 7
        neighbors[i].push_back(index_x_down + index_y_down*nl_x + index_z*nl_x*nl_y);
		// 8
        neighbors[i].push_back(index_x + index_y_down*nl_x + index_z*nl_x*nl_y);
		// 9
        neighbors[i].push_back(index_x_up + index_y_down*nl_x + index_z*nl_x*nl_y);
		// Above
		if(index_z_up < nl_z) {
			// 1
            neighbors[i].push_back(index_x_down + index_y_up*nl_x + index_z_up*nl_x*nl_y);
			// 2
            neighbors[i].push_back(index_x + index_y_up*nl_x + index_z_up*nl_x*nl_y);
			// 3
            neighbors[i].push_back(index_x_up + index_y_up*nl_x + index_z_up*nl_x*nl_y);
			// 4
            neighbors[i].push_back(index_x_down + index_y*nl_x + index_z_up*nl_x*nl_y);
			// 5
			neighbors[i].push_back(index_x + index_y*nl_x + index_z_up*nl_x*nl_y);
			// 6
            neighbors[i].push_back(index_x_up + index_y*nl_x + index_z_up*nl_x*nl_y);
			// 7
            neighbors[i].push_back(index_x_down + index_y_down*nl_x + index_z_up*nl_x*nl_y);
			// 8
            neighbors[i].push_back(index_x + index_y_down*nl_x + index_z_up*nl_x*nl_y);
			// 9
            neighbors[i].push_back(index_x_up + index_y_down*nl_x + index_z_up*nl_x*nl_y);
		}
	}
}

void NeighborList::GenerateCheckerboard(MembraneMC& sys) {    
	// Generates checkerboard list from current configuration
    // Note original Glotzer algorithm also performs cell shifts in one direction (x, -x, y, -y, z, and -z)
    // For now, pick random cell center location in (x,y) and build around that
    const int checkerboard_total_old = checkerboard_x*checkerboard_y;
    checkerboard_x = sys.lengths[0]/checkerboard_set_size;
    if(checkerboard_x%2==1) {
        checkerboard_x -= 1;
    }
    box_x_checkerboard = sys.lengths[0]/checkerboard_x;

    checkerboard_y = sys.lengths[1]/checkerboard_set_size;
    if(checkerboard_y%2==1) {
        checkerboard_y -= 1;
    }
    box_y_checkerboard = sys.lengths[1]/checkerboard_y;

	// Set up list
    const int checkerboard_total = checkerboard_x*checkerboard_y;
    vector<int> list(4.0*sys.vertices/double(checkerboard_total));
    // Evaluate number of dummies needed
    if(checkerboard_total != checkerboard_total_old) {
	    checkerboard_list.resize(checkerboard_total);
    }
    if(checkerboard_total_old == 1) {
        #pragma omp parallel for
        for(int i=0; i<checkerboard_total; i++) {
            checkerboard_list[i] = list;
        }
    }
    else if(checkerboard_total > checkerboard_total_old) {
        #pragma omp parallel for
        for(int i=checkerboard_total_old; i<checkerboard_total; i++) {
            checkerboard_list[i] = list;
        }
    }
    // Assign membrane particles to lists
    // Pick random cell center, build around that
    cell_center_x = box_x_checkerboard*sys.generator.d();
    cell_center_y = box_y_checkerboard*sys.generator.d();
	// Loop through membrane particles
    int index_particles[sys.vertices];
    int index_particles_add[sys.vertices];
    int index_particles_max[checkerboard_total];
    #pragma omp parallel for
    for(int i=0; i<checkerboard_total; i++) {
        index_particles_max[i] = 0;
    }
    #pragma omp parallel for
	for(int i=0; i<sys.vertices; i++) {
		// Determine index
		int index_x = floor((sys.lengths[0]*(sys.radii_tri[i][0]+0.5)-cell_center_x)/box_x_checkerboard);
		int index_y = floor((sys.lengths[1]*(sys.radii_tri[i][1]+0.5)-cell_center_y)/box_y_checkerboard);
        if(index_x == -1) {
            index_x += checkerboard_x;
        }
        if(index_y == -1) {
            index_y += checkerboard_y;
        }
        if(index_x == checkerboard_x) {
            index_x -= 1;
        }
        if(index_y == checkerboard_y) {
            index_y -= 1;
        }
        int particle_index = index_x + index_y*checkerboard_x;
		index_particles[i] = particle_index;
        #pragma omp atomic capture 
        index_particles_add[i] = index_particles_max[particle_index]++;
    }

    if(checkerboard_index.size() == 0) {
        checkerboard_index.resize(sys.vertices);
    }
    // First resize checkerboard_list in parallel
    #pragma omp parallel for
    for(int i=0; i<checkerboard_total; i++) {
        checkerboard_list[i].resize(index_particles_max[i]);
	}
    // Now add particles to checkerboard list in parallel
    // As index of adding particles is now, done trivially
    #pragma omp parallel for
    for(int i=0; i<sys.vertices; i++) {
		checkerboard_list[index_particles[i]][index_particles_add[i]] = i;
	}
    // Now add particles to checkerboard_idnex
    #pragma omp parallel for
    for(int i=0; i<sys.vertices; i++) {
		    checkerboard_index[i] = index_particles[i];
    }
}
