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
#include "neighborlist.hpp"
using namespace std;

void NeighborList::GenerateNeighborList() {    
	// Generates neighbor list from current configuration
	// vector<vector<vector<vector<double>>>> neighbor_list;
	// vector<vector<vector<vector<int>>>> neighbor_list_size;
    // Check to see if nl_x, nl_y are different
    // If yes, then continue, if not don't need to rebuild
    int nl_x_trial = int(Length_x)-1;
    int nl_y_trial = int(Length_y)-1;
    if(((nl_x_trial >= nl_x) && (nl_y_trial >= nl_y)) && ((nl_x_trial <= (nl_x+4)) && (nl_y_trial <= (nl_y+4)))) {
        return;
    }
    // Clear current lists
    /*
    cout << "Initial sizes" << endl;
    cout << neighbor_list.size() << endl;
    cout << neighbor_list_index.size() << endl;
    cout << neighbors.size() << endl;
    cout << "End sizes" << endl;
    */
    // CLEAR
    neighbor_list.clear();
    neighbor_list_index.clear();
    neighbors.clear();
    /*
    cout << "CLEAR" << endl;
    cout << neighbor_list.size() << endl;
    cout << neighbor_list_index.size() << endl;
    cout << neighbors.size() << endl;
    cout << "End sizes" << endl;
    */
	// Set up list
	vector<int> list;
    // Evaluate current values of nl_x, nl_y as that can change
    // Ideal box size is a little larger than 1.0. Get that converting
    // Length_x, Length_y to int then substracting by 1
    nl_x = int(Length_x)-2;
    nl_y = int(Length_y)-2;
    nl_z = int(Length_z*2.0)-1;
    // New box size values
    box_x = Length_x/double(nl_x); 
    box_y = Length_y/double(nl_y); 
    box_z = 2*Length_z/double(nl_z); 
    /*
    cout << Length_x << " " << nl_x << " " << box_x << endl;
    cout << Length_y << " " << nl_y << " " << box_y << endl;
    cout << Length_z << " " << nl_z << " " << box_z << endl;
    */
    // Evaluate number of dummies needed
	// cout << "nl_x nl_y nl_z " << nl_x << " " << nl_y << " " << nl_z << endl;
    
    neighbor_list_index.resize(vertices);
    neighbor_list.resize(nl_x*nl_y*nl_z);
    #pragma omp parallel for
	for(int i=0; i<nl_x*nl_y*nl_z; i++) {
    	neighbor_list[i] = list;
	}
	// Loop through membrane particles
    int index_particles[vertices];
    int index_particles_add[vertices];
    index_particles_max_nl.resize(nl_x*nl_y*nl_z,0);
    #pragma omp parallel for
	for(int i=0; i<vertices; i++) {
		// Determine index
		int index_x = int(Length_x*Radius_x_tri[i]/box_x);
		int index_y = int(Length_y*Radius_y_tri[i]/box_y);
		int index_z = int((Radius_z_tri[i]+Length_z)/box_z);
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
    for(int i=0; i<vertices; i++) {
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
    for(int i=0; i<vertices; i++) {
        neighbor_list[index_particles[i]][index_particles_add[i]] = i;
	}

    // cout << neighbor_list.size() << endl;
    // cout << neighbor_list_index.size() << endl;
	/*
	// Output neighbor list
	for(int i=0; i<nl_x*nl_y*nl_z; i++) {
		int index_x = i % nl_x;
		int index_y = (i % nl_x*nl_y)/nl_x;
		int index_z = i/(nl_x*nl_y);
		if(neighbor_list[i].size() != 0) {
			cout << "index_x " << index_x << " index_y " << index_y << " index_z " << index_z << endl;
			cout << "Particles ";
			for(int j=0; j<neighbor_list[i].size(); j++) {
				cout << neighbor_list[i][j] << " " << neighbor_list_index[neighbor_list[i][j]] << " ";
			}
			cout << endl;
		}
	}
	*/
	// Determine neighboring boxes
	vector<int> list_int;
	// cout << "Making neighbor list of neighbors" << endl;
	// cout << "nl_x " << nl_x << " nl_y " << nl_y << " nl_z " << nl_z << endl;
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
		// cout << "x " << index_x << " y " << index_y << " z " << index_z << endl;
		// cout << "Check " << i << " " << index_x+index_y*nl_x+index_z*nl_x*nl_y << endl;
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
    /*
	for(int i=0; i<nl_x*nl_y*nl_z; i++) {
		cout << i << " ";
		// Compare indices by backing out spatial coordinates and verifying
		int index_x = i % nl_x;
		int index_y = (i%(nl_x*nl_y))/nl_x;
		int index_z = i/(nl_x*nl_y);
		for(int j=0; j<neighbors[i].size(); j++) {
			cout << neighbors[i][j] << " ";
		}
		cout << endl;
	}
    */
    // cout << neighbors.size() << endl;
    // cout << neighbors[nl_x/2+nl_y/2*nl_x+nl_z/2*nl_x*nl_y].size() << endl;
    /*
    int test_particles = 0;
    for(int i=0; i<nl_x*nl_y*nl_z; i++) {
        test_particles += neighbor_list[i].size();
    }
    cout << "Number of particles in neighbor list " << test_particles << endl;
    */
}

void NeighborList::GenerateCheckerboard() {    
	// Generates checkerboard list from current configuration
    // Note original Glotzer algorithm also performs cell shifts in one direction (x, -x, y, -y, z, and -z)
    // For now, pick random cell center location in (x,y) and build around that
    // In any case, linear operation
    // More appropriate to pick one direction in GPU case to keep the memory more local
    // cout << Length_x/(2*checkerboard_set_size)*2 << endl;
    const int checkerboard_total_old = checkerboard_x*checkerboard_y;
    checkerboard_x = Length_x/checkerboard_set_size;
    if(checkerboard_x%2==1) {
        checkerboard_x -= 1;
    }
    box_x_checkerboard = Length_x/checkerboard_x;
    checkerboard_y = Length_y/checkerboard_set_size;
    if(checkerboard_y%2==1) {
        checkerboard_y -= 1;
    }
    box_y_checkerboard = Length_y/checkerboard_y;
    // Test to make sure this is right. Might have to have round/nearbyint used
    // Clear current lists
    /*
    cout << "Initial sizes" << endl;
    cout << checkerboard_list.size() << endl;
    cout << checkerboard_list_index.size() << endl;
    cout << checkerboards.size() << endl;
    cout << "End sizes" << endl;
    */
    // CLEAR
    // checkerboard_list.clear();
    // checkerboard_index.clear();
    // checkerboard_neighbors.clear();
    /*
    cout << "CLEAR" << endl;
    cout << checkerboard_list.size() << endl;
    cout << checkerboard_list_index.size() << endl;
    cout << checkerboards.size() << endl;
    cout << "End sizes" << endl;
    */
	// Set up list
    const int checkerboard_total = checkerboard_x*checkerboard_y;
    vector<int> list(4.0*vertices/double(checkerboard_total));
    // Evaluate number of dummies needed
	// cout << "nl_x nl_y nl_z " << nl_x << " " << nl_y << " " << nl_z << endl;
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
    cell_center_x = box_x_checkerboard*generator.d();
    cell_center_y = box_y_checkerboard*generator.d();
    // cout << cell_center_x << " " << cell_center_y << endl;
    // cout << box_x_checkerboard << " " << box_y_checkerboard << endl;
	// Loop through membrane particles
    int index_particles[vertices];
    int index_particles_add[vertices];
    int index_particles_max[checkerboard_total];
    #pragma omp parallel for
    for(int i=0; i<checkerboard_total; i++) {
        index_particles_max[i] = 0;
    }
    #pragma omp parallel for
	for(int i=0; i<vertices; i++) {
		// Determine index
		int index_x = floor((Length_x*Radius_x_tri[i]-cell_center_x)/box_x_checkerboard);
		int index_y = floor((Length_y*Radius_y_tri[i]-cell_center_y)/box_y_checkerboard);
        // cout << index_x << " " << index_y << endl;
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
        // cout << "After " << index_x << " " << index_y << endl;
        int particle_index = index_x + index_y*checkerboard_x;
		index_particles[i] = particle_index;
        #pragma omp atomic capture 
        index_particles_add[i] = index_particles_max[particle_index]++;
        // cout << index << endl;
    }

    if(checkerboard_index.size() == 0) {
        checkerboard_index.resize(vertices);
    }
    // First resize checkerboard_list in parallel
    #pragma omp parallel for
    for(int i=0; i<checkerboard_total; i++) {
        checkerboard_list[i].resize(index_particles_max[i]);
	}
    // Now add particles to checkerboard list in parallel
    // As index of adding particles is now, done trivially
    #pragma omp parallel for
    for(int i=0; i<vertices; i++) {
		checkerboard_list[index_particles[i]][index_particles_add[i]] = i;
	}
    // Now add particles to checkerboard_idnex
    #pragma omp parallel for
    for(int i=0; i<vertices; i++) {
		    checkerboard_index[i] = index_particles[i];
    }
    // Shuffle checkerboard_list
    /*
    #pragma omp parallel for
    for(int i=0; i<checkerboard_x*checkerboard_y; i++) {
        Saru& local_generator = generators[omp_get_thread_num()];
        shuffle_saru(local_generator, checkerboard_list[i]);
    }
    */
    // cout << checkerboard_list.size() << endl;
    // cout << checkerboard_index.size() << endl;
	/*
	// Output checkerboard list
	for(int i=0; i<checkerboard_x*checkerboard_y; i++) {
		int index_x = i % checkerboard_x;
		int index_y = i/checkerboard_x;
		if(checkerboard_list[i].size() != 0) {
			cout << "index_x " << index_x << " index_y " << index_y << endl;
			cout << "Particles ";
			for(int j=0; j<checkerboard_list[i].size(); j++) {
				cout << checkerboard_list[i][j] << " " << checkerboard_list_index[checkerboard_list[i][j]] << " ";
			}
			cout << endl;
		}
	}
	*/
	// Determine neighboring boxes
	// cout << "Making neighbor list of neighbors" << endl;
	// cout << "nl_x " << nl_x << " nl_y " << nl_y << " nl_z " << nl_z << endl;
    /*
    for(int i=0; i<checkerboard_x*checkerboard_y; i++) {
		checkerboard_neighbors.push_back(list_int);
		int index_x = i % checkerboard_x;
		int index_y = i/checkerboard_x;
		// Sweep over neighbors
		int index_y_down = ((index_y-1)%checkerboard_y+checkerboard_y)%checkerboard_y;
		int index_y_up = ((index_y+1)%checkerboard_y+checkerboard_y)%checkerboard_y;
		int index_x_down = ((index_x-1)%checkerboard_x+checkerboard_x)%checkerboard_x;
		int index_x_up = ((index_x+1)%checkerboard_x+checkerboard_x)%checkerboard_x;
		// cout << "x " << index_x << " y " << index_y << endl;
		// cout << "Check " << i << " " << index_x+index_y*checkerboard_x << endl;
		// Checking below, middle, and above cases
		// For each one, xy stencil check will be in this order
		// 1 2 3
		// 4 5 6
		// 7 8 9
		// Below
		// 1
        checkerboard_neighbors[i].push_back(index_x_down + index_y_up*checkerboard_x);
		// 2
        checkerboard_neighbors[i].push_back(index_x + index_y_up*checkerboard_x);
		// 3
        checkerboard_neighbors[i].push_back(index_x_up + index_y_up*checkerboard_x);
		// 4
        checkerboard_neighbors[i].push_back(index_x_down + index_y*checkerboard_x);
		// 5
		checkerboard_neighbors[i].push_back(index_x + index_y*checkerboard_x);
		// 6
        checkerboard_neighbors[i].push_back(index_x_up + index_y*checkerboard_x);
		// 7
        checkerboard_neighbors[i].push_back(index_x_down + index_y_down*checkerboard_x);
		// 8
        checkerboard_neighbors[i].push_back(index_x + index_y_down*checkerboard_x);
		// 9
        checkerboard_neighbors[i].push_back(index_x_up + index_y_down*checkerboard_x);
	}
    */
    /*
	for(int i=0; i<nl_x*nl_y*nl_z; i++) {
		cout << i << " ";
		// Compare indices by backing out spatial coordinates and verifying
		int index_x = i % nl_x;
		int index_y = (i%(nl_x*nl_y))/nl_x;
		int index_z = i/(nl_x*nl_y);
		for(int j=0; j<neighbors[i].size(); j++) {
			cout << neighbors[i][j] << " ";
		}
		cout << endl;
	}
    */
    // cout << neighbors.size() << endl;
    // cout << neighbors[nl_x/2+nl_y/2*nl_x+nl_z/2*nl_x*nl_y].size() << endl;
    /*
    int test_particles = 0;
    for(int i=0; i<nl_x*nl_y*nl_z; i++) {
        test_particles += neighbor_list[i].size();
    }
    cout << "Number of particles in neighbor list " << test_particles << endl;
    */
}
