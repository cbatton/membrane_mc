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
#include "init_system.hpp"
#include "utilities.hpp"
#include "saruprng.hpp"
using namespace std;

InitSystem::InitSystem() {
    // Constructor
    // Does nothing
}

InitSystem::~InitSystem() {
    // Destructor
    // Does nothing
}

void InitSystem::Initialize(MembraneMC& sys) {
    // Use initializer to handle system initialization
    Utilities util;
    util.SaruSeed(sys,sys.count_step);
    InitializeEquilState(sys);
    GenerateTriangulationEquil(sys);
    UseTriangulation(sys,"out.off");
    // now place proteins
    #pragma omp parallel for
    for(int i=0; i<sys.vertices; i++) {
        sys.protein_node[i] = -1;
    }
    // Place proteins at random
    for(int i=0; i<sys.num_proteins; i++) {
        bool check_nodes = true;
        while(check_nodes) {
            int j = sys.generator.rand_select(sys.vertices-1);
            if(sys.ising_array[j] != 2) {
                check_nodes = false;
                sys.ising_array[j] = 2;
                sys.protein_node[j] = 0;
            }
        }
    }
    // End clock
    sys.t2 = chrono::steady_clock::now();
    chrono::duration<double> time_span_init = sys.t2-sys.t1;
    sys.time_storage_other[0] += time_span_init.count();
}

void InitSystem::InitializeEquilState(MembraneMC& sys) {
    // Create two layers of points that will triangulate to form a mesh of equilateral triangle
    // Layer one
    for(int i=0; i<sys.dim_x; i += 2){
        for(int j=0; j<sys.dim_y; j++) {
            sys.radii_tri[j+i*sys.dim_y][0] = double(sys.lengths[0]*(i+0.5)/sys.dim_x)/sys.lengths[0]-0.5;
            sys.radii_tri[j+i*sys.dim_y][1] = double(sys.lengths[1]*(j+0.25)/sys.dim_y)/sys.lengths[1]-0.5;
            sys.radii_tri[j+i*sys.dim_y][2] = 0.0;
            sys.radii_tri_original[j+i*sys.dim_y] = sys.radii_tri[j+i*sys.dim_y];
        }
    }
    for(int i=1; i<sys.dim_x; i += 2){
        for(int j=0; j<sys.dim_y; j++) {
            sys.radii_tri[j+i*sys.dim_y][0] = double(sys.lengths[0]*(i+0.5)/sys.dim_x)/sys.lengths[0]-0.5;
            sys.radii_tri[j+i*sys.dim_y][1] = double(sys.lengths[1]*(j+0.75)/sys.dim_y)/sys.lengths[1]-0.5;
            sys.radii_tri[j+i*sys.dim_y][2] = 0.0;
            sys.radii_tri_original[j+i*sys.dim_y] = sys.radii_tri[j+i*sys.dim_y];
        }
    }
    
    for(int i=0; i<(sys.vertices*sys.num_frac); i++) {
		int j = sys.generator.rand_select(sys.vertices-1);
		while (sys.ising_array[j] == 1) {
			j = sys.generator.rand_select(sys.vertices-1);
        }
        sys.ising_array[j] = 1;
    }    
}

void InitSystem::GenerateTriangulationEquil(MembraneMC& sys) {
    // Output initial configuration as an off file
	ofstream myfile;
	myfile.open(sys.output_path+"/out.off");
	myfile << 0 << " " << 0 << " " << sys.lengths[0] << " " << sys.lengths[1] << endl;
    myfile << 1 << " " << 1 << endl;
    myfile << sys.vertices << endl;
    for(int i=0; i<sys.dim_x; i++){
		for(int j=0; j<sys.dim_y; j++){	
			myfile << sys.lengths[0]*sys.radii_tri[j+i*sys.dim_y][0] << " " << sys.lengths[1]*sys.radii_tri[j+i*sys.dim_y][1] << endl;
		}
	}
	myfile << endl;
	myfile << sys.faces << endl;
    const int faces_active = sys.faces;
	int triangle_list_active[faces_active][3];
	int face_count = 0;
    // Add triangles in body
    for(int i=0; i<sys.dim_x; i++){
        for(int j=0; j<sys.dim_y; j++){
            if(i%2==0) {
				int dummy_up = j+1;
				if(dummy_up >= sys.dim_y) {
					dummy_up -= sys.dim_y;
				}
				int dummy_down = j-1;
				if(dummy_down < 0) {
					dummy_down += sys.dim_y;
				}
				int dummy_right = i+1;
				if(dummy_right >= sys.dim_x) {
					dummy_right -= sys.dim_x;
				}
				int dummy_left = i-1;
				if(dummy_left < 0) {
					dummy_left += sys.dim_x;
				}
                // Basic idea here is to form the six triangles around each point, and then check to see if they are on the active triangle list
                // If not, add to active triangle list and write to myfile
                // New idea: generate six neighboring points, attempt to add to 
                int triangle_trials[6][3];
                triangle_trials[0][0] = j+i*sys.dim_y; triangle_trials[0][1] = dummy_up+i*sys.dim_y; triangle_trials[0][2] = j+dummy_right*sys.dim_y;
                triangle_trials[1][0] = j+i*sys.dim_y; triangle_trials[1][1] = dummy_down+dummy_right*sys.dim_y; triangle_trials[1][2] = j+dummy_right*sys.dim_y;
                triangle_trials[2][0] = j+i*sys.dim_y; triangle_trials[2][1] = dummy_down+i*sys.dim_y; triangle_trials[2][2] = dummy_down+dummy_right*sys.dim_y;
                triangle_trials[3][0] = j+i*sys.dim_y; triangle_trials[3][1] = dummy_up+i*sys.dim_y; triangle_trials[3][2] = j+dummy_left*sys.dim_y;
                triangle_trials[4][0] = j+i*sys.dim_y; triangle_trials[4][1] = dummy_down+dummy_left*sys.dim_y; triangle_trials[4][2] = j+dummy_left*sys.dim_y;
                triangle_trials[5][0] = j+i*sys.dim_y; triangle_trials[5][1] = dummy_down+i*sys.dim_y; triangle_trials[5][2] = dummy_down+dummy_left*sys.dim_y;
                // Attempt to add triangles to triangle_list_active. If not a duplicate, print to file
                for(int k=0; k<6; k++) {
                    bool Replica = false;
                    for(int l=0; l<face_count; l++) {
                        if((triangle_list_active[l][0] == triangle_trials[k][0]) && (triangle_list_active[l][1] == triangle_trials[k][1]) && (triangle_list_active[l][2] == triangle_trials[k][2])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][0]) && (triangle_list_active[l][1] == triangle_trials[k][2]) && (triangle_list_active[l][2] == triangle_trials[k][1])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][1]) && (triangle_list_active[l][1] == triangle_trials[k][0]) && (triangle_list_active[l][2] == triangle_trials[k][2])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][1]) && (triangle_list_active[l][1] == triangle_trials[k][2]) && (triangle_list_active[l][2] == triangle_trials[k][0])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][2]) && (triangle_list_active[l][1] == triangle_trials[k][1]) && (triangle_list_active[l][2] == triangle_trials[k][0])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][2]) && (triangle_list_active[l][1] == triangle_trials[k][0]) && (triangle_list_active[l][2] == triangle_trials[k][1])) {
                            Replica = true;
                        }
                    }
                    if(!Replica) {
                        triangle_list_active[face_count][0] = triangle_trials[k][0];
                        triangle_list_active[face_count][1] = triangle_trials[k][1];
                        triangle_list_active[face_count][2] = triangle_trials[k][2];
                        face_count++;
                        myfile << triangle_trials[k][0] << " " << triangle_trials[k][1] << " " << triangle_trials[k][2] << endl;
                    }
                }
            }
            else if(i%2==1) {
				int dummy_up = j+1;
				if(dummy_up >= sys.dim_y) {
					dummy_up -= sys.dim_y;
				}
				int dummy_down = j-1;
				if(dummy_down < 0) {
					dummy_down += sys.dim_y;
				}
				int dummy_right = i+1;
				if(dummy_right >= sys.dim_x) {
					dummy_right -= sys.dim_x;
				}
				int dummy_left = i-1;
				if(dummy_left < 0) {
					dummy_left += sys.dim_x;
				}
                // Basic idea here is to form the six triangles around each point, and then check to see if they are on the active triangle list
                // If not, add to active triangle list and write to myfile
                // New idea: generate six neighboring points, attempt to add to 
                int triangle_trials[6][3];
                triangle_trials[0][0] = j+i*sys.dim_y; triangle_trials[0][1] = dummy_up+i*sys.dim_y; triangle_trials[0][2] = dummy_up+dummy_right*sys.dim_y;
                triangle_trials[1][0] = j+i*sys.dim_y; triangle_trials[1][1] = dummy_up+dummy_right*sys.dim_y; triangle_trials[1][2] = j+dummy_right*sys.dim_y;
                triangle_trials[2][0] = j+i*sys.dim_y; triangle_trials[2][1] = dummy_down+i*sys.dim_y; triangle_trials[2][2] = j+dummy_right*sys.dim_y;
                triangle_trials[3][0] = j+i*sys.dim_y; triangle_trials[3][1] = dummy_up+i*sys.dim_y; triangle_trials[3][2] = dummy_up+dummy_left*sys.dim_y;
                triangle_trials[4][0] = j+i*sys.dim_y; triangle_trials[4][1] = dummy_up+dummy_left*sys.dim_y; triangle_trials[4][2] = j+dummy_left*sys.dim_y;
                triangle_trials[5][0] = j+i*sys.dim_y; triangle_trials[5][1] = dummy_down+i*sys.dim_y; triangle_trials[5][2] = j+dummy_left*sys.dim_y;
                // Attempt to add triangles to triangle_list_active. If not a duplicate, print to file
                for(int k=0; k<6; k++) {
                    bool Replica = false;
                    for(int l=0; l<face_count; l++) {
                        if((triangle_list_active[l][0] == triangle_trials[k][0]) && (triangle_list_active[l][1] == triangle_trials[k][1]) && (triangle_list_active[l][2] == triangle_trials[k][2])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][0]) && (triangle_list_active[l][1] == triangle_trials[k][2]) && (triangle_list_active[l][2] == triangle_trials[k][1])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][1]) && (triangle_list_active[l][1] == triangle_trials[k][0]) && (triangle_list_active[l][2] == triangle_trials[k][2])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][1]) && (triangle_list_active[l][1] == triangle_trials[k][2]) && (triangle_list_active[l][2] == triangle_trials[k][0])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][2]) && (triangle_list_active[l][1] == triangle_trials[k][1]) && (triangle_list_active[l][2] == triangle_trials[k][0])) {
                            Replica = true;
                        }
                        else if((triangle_list_active[l][0] == triangle_trials[k][2]) && (triangle_list_active[l][1] == triangle_trials[k][0]) && (triangle_list_active[l][2] == triangle_trials[k][1])) {
                            Replica = true;
                        }
                    }
                    if(!Replica) {
                        triangle_list_active[face_count][0] = triangle_trials[k][0];
                        triangle_list_active[face_count][1] = triangle_trials[k][1];
                        triangle_list_active[face_count][2] = triangle_trials[k][2];
                        face_count++;
                        myfile << triangle_trials[k][0] << " " << triangle_trials[k][1] << " " << triangle_trials[k][2] << endl;
                    }
                }
            } 
        }
    }
	myfile.close();
}

inline int InitSystem::LinkTriangleTest(MembraneMC& sys, int vertex_1, int vertex_2) {
// Determines if two points are linked together
    int link_1_2 = 0;
    for(int i=0; i<sys.point_neighbor_list[vertex_1].size(); i++) {
        if(sys.point_neighbor_list[vertex_1][i] == vertex_2) {
            link_1_2 += 1;
            break;
        }
    }
    return link_1_2;
}

void InitSystem::UseTriangulation(MembraneMC& sys, string name) {
    ifstream input;
    Utilities util;
    input.open(sys.output_path+"/"+name);
    vector<int> point_neighbor_list_m(sys.vertices, 0);
    vector<int> point_triangle_list_m(sys.vertices, 0);
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        sys.my_cout << "No input file." << endl;
    }

    else{
        string line;
        // Skip 3 lines
        for(int i=0; i<3; i++){
            getline(input,line);
        }

        // Input radius values
        for(int i=0; i<sys.vertices; i++){
            input >> sys.radii_tri[i][0] >> sys.radii_tri[i][1];
            sys.radii_tri[i][2] = 0;
            sys.radii_tri[i][0] = sys.radii_tri[i][0]/sys.lengths[0];
            sys.radii_tri[i][1] = sys.radii_tri[i][1]/sys.lengths[1];
			// sys.my_cout << "x y z " << sys.lengths[0]*sys.radii_tri[i][0] << " " << sys.lengths[1]*sys.radii_tri[i][1] << " " << sys.radii_tri[i][2] << endl;
            sys.radii_tri_original[i] = sys.radii_tri[i];
            getline(input,line);
        }

        // Skip 2 lines
        for(int i=0; i<2; i++){
            getline(input,line);
        } 
        // Input triangle connectivity
        // This process can be simultaneously done with generating point neighbor list
        // Initialize point_neighbor_list to null values
        // Okay, clever way to do this stuff
        // Replace negative values in point_neighbor_list with actual positive values
        // Such that skip already inputted positive values, add at nearest negative number
        // Process will naturally double count links in neighbor list
        // In while loop skipping negative numbers, if to be added value is unique add to first entry of point_neighbor_triangle
        // In while loop skipping negative numbers, if to be added value is the same as already in place value add to second entry of point_neighbor_triangle
        // point_neighbor_triangle is needed to compute needed cotan angles by evaluating angle through standard formula with other links in triangle
        for(int i=0; i<sys.faces; i++){
            input >> sys.triangle_list[i][0] >> sys.triangle_list[i][1] >> sys.triangle_list[i][2];
            // Keep orientation consistent here
            // Initial basis is all normals pointing up in z-direction
            double normal_test[3] = {0,0,0};
            util.NormalTriangle(sys, i, normal_test);
            if(normal_test[2] < 0) {
                // If so, swapping first two entries will make normals okay
                int triangle_list_0 = sys.triangle_list[i][0];
                sys.triangle_list[i][0] = sys.triangle_list[i][1];
                sys.triangle_list[i][1] = triangle_list_0;
            }
            // sys.my_cout << "For face " << i << " the points are " << sys.triangle_list[i][0] << " " << sys.triangle_list[i][1] << " " << sys.triangle_list[i][2] << endl;
            // Most likely don't need this for loop, just need to add connections from 0->1, 1->2, 2->1
            int place_holder_nl = 0;
            // Add to point_triangle_list
            if (sys.point_triangle_list[sys.triangle_list[i][0]][0] == -1) {
                sys.point_triangle_list[sys.triangle_list[i][0]][0] = i;
                point_triangle_list_m[sys.triangle_list[i][0]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < sys.neighbor_max) {
                    if(sys.point_triangle_list[sys.triangle_list[i][0]][place_holder_nl] == -1) {
                        sys.point_triangle_list[sys.triangle_list[i][0]][place_holder_nl] = i;
                        place_holder_nl = sys.neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_list_m[sys.triangle_list[i][0]] += 1;
            }
            if (sys.point_triangle_list[sys.triangle_list[i][1]][0] == -1) {
                sys.point_triangle_list[sys.triangle_list[i][1]][0] = i;
                point_triangle_list_m[sys.triangle_list[i][1]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < sys.neighbor_max) {
                    if(sys.point_triangle_list[sys.triangle_list[i][1]][place_holder_nl] == -1) {
                        sys.point_triangle_list[sys.triangle_list[i][1]][place_holder_nl] = i;
                        place_holder_nl = sys.neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_list_m[sys.triangle_list[i][1]] += 1;
            }
            if (sys.point_triangle_list[sys.triangle_list[i][2]][0] == -1) {
                sys.point_triangle_list[sys.triangle_list[i][2]][0] = i;
                point_triangle_list_m[sys.triangle_list[i][2]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < sys.neighbor_max) {
                    if(sys.point_triangle_list[sys.triangle_list[i][2]][place_holder_nl] == -1) {
                        sys.point_triangle_list[sys.triangle_list[i][2]][place_holder_nl] = i;
                        place_holder_nl = sys.neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_list_m[sys.triangle_list[i][2]] += 1;
            }
            // For loop variable was j. Now, idea is the following
            // Sort entries of triangle_list so we have low, medium, high
            // Do link list as doing that by [lower][other] on the vertex, so do that for low -> medium, low -> high, medium -> high
            // Thus go in order of low (most steps), medium (less steps), high (just add to point_neighbor_list)
            int low_tri;
            int med_tri;
            int high_tri;
            if ((sys.triangle_list[i][0] > sys.triangle_list[i][1]) && (sys.triangle_list[i][1] > sys.triangle_list[i][2])) {
                high_tri = sys.triangle_list[i][0];
                med_tri = sys.triangle_list[i][1];
                low_tri = sys.triangle_list[i][2];
                // sys.my_cout << "high med low" << endl;
            }
            else if((sys.triangle_list[i][1] > sys.triangle_list[i][0]) && (sys.triangle_list[i][0] > sys.triangle_list[i][2])) {
                high_tri = sys.triangle_list[i][1];
                med_tri = sys.triangle_list[i][0];
                low_tri = sys.triangle_list[i][2];
                // sys.my_cout << "med high low" << endl;
            }
            else if((sys.triangle_list[i][2] > sys.triangle_list[i][0]) && (sys.triangle_list[i][0] > sys.triangle_list[i][1])) {
                high_tri = sys.triangle_list[i][2];
                med_tri = sys.triangle_list[i][0];
                low_tri = sys.triangle_list[i][1];
                // sys.my_cout << "med low high" << endl;
            }
            else if((sys.triangle_list[i][2] > sys.triangle_list[i][1]) && (sys.triangle_list[i][1] > sys.triangle_list[i][0])) {
                high_tri = sys.triangle_list[i][2];
                med_tri = sys.triangle_list[i][1];
                low_tri = sys.triangle_list[i][0];
                // sys.my_cout << "low med high" << endl;
            }
            else if((sys.triangle_list[i][0] > sys.triangle_list[i][2]) && (sys.triangle_list[i][2] > sys.triangle_list[i][1])) {
                high_tri = sys.triangle_list[i][0];
                med_tri = sys.triangle_list[i][2];
                low_tri = sys.triangle_list[i][1];
                // sys.my_cout << "high low med" << endl;
            }
            else {
                high_tri = sys.triangle_list[i][1];
                med_tri = sys.triangle_list[i][2];
                low_tri = sys.triangle_list[i][0];
                // sys.my_cout << "low high med" << endl;
            }

            // Note have to do
            // low -> med, low -> high, med -> low, high -> low, med -> high, high -> med
            // 
            // This case is low -> med, low -> high and reverse cases
            // sys.my_cout << "Now onto the sorting hat" << endl;
            if(sys.point_neighbor_list[low_tri][0] == -1){
                // Generalize as triangle list second value is not necessarily 1,2
                // sys.my_cout << "No neighbors" << endl;
                sys.point_neighbor_list[low_tri][0] = med_tri;
                sys.point_neighbor_triangle[low_tri][0][0] = i;
                point_neighbor_list_m[low_tri] += 2;
                // sys.my_cout << "Neighbor list at " << low_tri << " is now " << sys.point_neighbor_list[low_tri][0] << endl;
                // Check opposite direction
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                        sys.point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        // sys.my_cout << "Neighbor list at " << med_tri << " entry " << place_holder_nl <<  " is now " << sys.point_neighbor_list[med_tri][place_holder_nl] << endl;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[med_tri] += 1;
    
                sys.point_neighbor_list[low_tri][1] = high_tri;
                sys.point_neighbor_triangle[low_tri][1][0] = i;
                // sys.my_cout << "Neighbor list at " << low_tri << " is now " << sys.point_neighbor_list[low_tri][1] << endl;
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                        sys.point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        // sys.my_cout << "Neighbor list at " << high_tri << " entry " << place_holder_nl <<  " is now " << sys.point_neighbor_list[high_tri][place_holder_nl] << endl;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[high_tri] += 1;
            }
            // Make big else loop in other case 
            else {
                // sys.my_cout << "Are neighbors initially" << endl;
                // Check to see if low_tri and med_tri are linked
                int link_low_med = LinkTriangleTest(sys, low_tri, med_tri);
                // Convert statements to else as there is no else if if it is not not -1
                if(link_low_med != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[low_tri][place_holder_nl] = med_tri;
                            sys.point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                            sys.point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[med_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_list_m[low_tri]; j++) {
                        if(sys.point_neighbor_list[low_tri][j] == med_tri) {
                            sys.point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_list_m[med_tri]; j++) {
                        if(sys.point_neighbor_list[med_tri][j] == low_tri) {
                            sys.point_neighbor_triangle[med_tri][j][1] = i;
                            break;
                        }
                    }
                }

                int link_low_high = LinkTriangleTest(sys, low_tri, high_tri);
                if(link_low_high != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[low_tri][place_holder_nl] = high_tri;
                            sys.point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                            sys.point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[high_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_list_m[low_tri]; j++) {
                        if(sys.point_neighbor_list[low_tri][j] == high_tri) {
                            sys.point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_list_m[high_tri]; j++) {
                        if(sys.point_neighbor_list[high_tri][j] == low_tri) {
                            sys.point_neighbor_triangle[high_tri][j][1] = i;
                            break;
                        }
                    }
                }

            }
            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
                
            // Can use these values to
            // This case is med -> high and reverse cases
            int link_med_high = LinkTriangleTest(sys, med_tri, high_tri);
            if(link_med_high != 1) {
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[med_tri][place_holder_nl] = high_tri;
                        sys.point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[med_tri] += 1;
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[high_tri][place_holder_nl] = med_tri;
                        sys.point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[high_tri] += 1;
            }
            // If else, then we've seen this link before
            // Add to second point_neighbor_triangle entry
            else {
                for(int j=0; j<point_neighbor_list_m[med_tri]; j++) {
                    if(sys.point_neighbor_list[med_tri][j] == high_tri) {
                        sys.point_neighbor_triangle[med_tri][j][1] = i;
                        break;
                    }
                }
                for(int j=0; j<point_neighbor_list_m[high_tri]; j++) {
                    if(sys.point_neighbor_list[high_tri][j] == med_tri) {
                        sys.point_neighbor_triangle[high_tri][j][1] = i;
                        break;
                    }
                }
            }

            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
            getline(input,line);
        }
    }

    // Now resize all vector entries
    for(int i=0; i<sys.vertices; i++) {
        for(int j=0; j<sys.neighbor_max; j++) {
            if(sys.point_neighbor_list[i][j] == -1) {
                sys.point_neighbor_list[i].resize(j);
                sys.point_neighbor_triangle[i].resize(j);
                break;
            }
            if(sys.point_triangle_list[i][j] == -1) {
                sys.point_triangle_list[i].resize(j);
            }
        }
        for(int j=0; j<sys.neighbor_max; j++) {
            if(sys.point_triangle_list[i][j] == -1) {
                sys.point_triangle_list[i].resize(j);
                break;
            }
        }
    }
    // Set reference original values equal to current entries
    sys.triangle_list_original = sys.triangle_list;
    sys.point_neighbor_list_original = sys.point_neighbor_list;
    sys.point_triangle_list_original = sys.point_triangle_list;
    sys.point_neighbor_triangle_original = sys.point_neighbor_triangle;
}

void InitSystem::UseTriangulationRestart(MembraneMC& sys, string name) {
    ifstream input;
    input.open(sys.output_path+"/"+name);
    vector<int> point_neighbor_list_m(sys.vertices, 0);
    vector<int> point_triangle_list_m(sys.vertices, 0);
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        sys.my_cout << "No input file." << endl;
    }

    else{
        string line;
        // sys.my_cout << "Connectivity file detected. Changing values." << endl;
        // Get box size
        input >> line >> sys.lengths[0] >> line >> sys.lengths[1] >> line >> sys.lengths[2] >> line >> sys.count_step;
        sys.scale_xy = sys.lengths[0]/sys.lengths_base[0];
        sys.lengths_old = sys.lengths;
        // Skip 2 lines
        for(int i=0; i<2; i++){
            getline(input,line);
        } 
        // Input radius values
        for(int i=0; i<sys.vertices; i++){
            input >> sys.ising_array[i] >> sys.radii_tri[i][0] >> sys.radii_tri[i][1] >> sys.radii_tri[i][2];
            if(sys.ising_array[i] == 2) {
                sys.protein_node[i] = 0;
            }
            sys.radii_tri[i][0] = sys.radii_tri[i][0]/sys.lengths[0];
            sys.radii_tri[i][1] = sys.radii_tri[i][1]/sys.lengths[1];
			// sys.my_cout << "x y z " << sys.lengths[0]*sys.radii_tri[i][0] << " " << sys.lengths[1]*sys.radii_tri[i][1] << " " << sys.radii_tri[i][2] << endl;
            sys.radii_tri_original[i] = sys.radii_tri[i];
            getline(input,line);
        }

        // Skip 2 lines
        for(int i=0; i<2; i++){
            getline(input,line);
        } 
        // Input triangle connectivity
        // This process can be simultaneously done with generating point neighbor list
        // Initialize point_neighbor_list to null values
        // Okay, clever way to do this stuff
        // Replace negative values in point_neighbor_list with actual positive values
        // Such that skip already inputted positive values, add at nearest negative number
        // Process will naturally double count links in neighbor list
        // In while loop skipping negative numbers, if to be added value is unique add to first entry of point_neighbor_triangle
        // In while loop skipping negative numbers, if to be added value is the same as already in place value add to second entry of point_neighbor_triangle
        // point_neighbor_triangle is needed to compute needed cotan angles by evaluating angle through standard formula with other links in triangle
        for(int i=0; i<sys.faces; i++){
            input >> sys.triangle_list[i][0] >> sys.triangle_list[i][1] >> sys.triangle_list[i][2];
            // sys.my_cout << "For face " << i << " the points are " << sys.triangle_list[i][0] << " " << sys.triangle_list[i][1] << " " << sys.triangle_list[i][2] << endl;
            int place_holder_nl = 0;
            // Add to point_triangle_list
            if (sys.point_triangle_list[sys.triangle_list[i][0]][0] == -1) {
                sys.point_triangle_list[sys.triangle_list[i][0]][0] = i;
                point_triangle_list_m[sys.triangle_list[i][0]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < sys.neighbor_max) {
                    if(sys.point_triangle_list[sys.triangle_list[i][0]][place_holder_nl] == -1) {
                        sys.point_triangle_list[sys.triangle_list[i][0]][place_holder_nl] = i;
                        place_holder_nl = sys.neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_list_m[sys.triangle_list[i][0]] += 1;
            }
            if (sys.point_triangle_list[sys.triangle_list[i][1]][0] == -1) {
                sys.point_triangle_list[sys.triangle_list[i][1]][0] = i;
                point_triangle_list_m[sys.triangle_list[i][1]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < sys.neighbor_max) {
                    if(sys.point_triangle_list[sys.triangle_list[i][1]][place_holder_nl] == -1) {
                        sys.point_triangle_list[sys.triangle_list[i][1]][place_holder_nl] = i;
                        place_holder_nl = sys.neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_list_m[sys.triangle_list[i][1]] += 1;
            }
            if (sys.point_triangle_list[sys.triangle_list[i][2]][0] == -1) {
                sys.point_triangle_list[sys.triangle_list[i][2]][0] = i;
                point_triangle_list_m[sys.triangle_list[i][2]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < sys.neighbor_max) {
                    if(sys.point_triangle_list[sys.triangle_list[i][2]][place_holder_nl] == -1) {
                        sys.point_triangle_list[sys.triangle_list[i][2]][place_holder_nl] = i;
                        place_holder_nl = sys.neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_list_m[sys.triangle_list[i][2]] += 1;
            }
            // For loop variable was j. Now, idea is the following
            // Sort entries of triangle_list so we have low, medium, high
            // Do link list as doing that by [lower][other] on the vertex, so do that for low -> medium, low -> high, medium -> high
            // Thus go in order of low (most steps), medium (less steps), high (just add to point_neighbor_list)
            int low_tri;
            int med_tri;
            int high_tri;
            if ((sys.triangle_list[i][0] > sys.triangle_list[i][1]) && (sys.triangle_list[i][1] > sys.triangle_list[i][2])) {
                high_tri = sys.triangle_list[i][0];
                med_tri = sys.triangle_list[i][1];
                low_tri = sys.triangle_list[i][2];
                // sys.my_cout << "high med low" << endl;
            }
            else if((sys.triangle_list[i][1] > sys.triangle_list[i][0]) && (sys.triangle_list[i][0] > sys.triangle_list[i][2])) {
                high_tri = sys.triangle_list[i][1];
                med_tri = sys.triangle_list[i][0];
                low_tri = sys.triangle_list[i][2];
                // sys.my_cout << "med high low" << endl;
            }
            else if((sys.triangle_list[i][2] > sys.triangle_list[i][0]) && (sys.triangle_list[i][0] > sys.triangle_list[i][1])) {
                high_tri = sys.triangle_list[i][2];
                med_tri = sys.triangle_list[i][0];
                low_tri = sys.triangle_list[i][1];
                // sys.my_cout << "med low high" << endl;
            }
            else if((sys.triangle_list[i][2] > sys.triangle_list[i][1]) && (sys.triangle_list[i][1] > sys.triangle_list[i][0])) {
                high_tri = sys.triangle_list[i][2];
                med_tri = sys.triangle_list[i][1];
                low_tri = sys.triangle_list[i][0];
                // sys.my_cout << "low med high" << endl;
            }
            else if((sys.triangle_list[i][0] > sys.triangle_list[i][2]) && (sys.triangle_list[i][2] > sys.triangle_list[i][1])) {
                high_tri = sys.triangle_list[i][0];
                med_tri = sys.triangle_list[i][2];
                low_tri = sys.triangle_list[i][1];
                // sys.my_cout << "high low med" << endl;
            }
            else {
                high_tri = sys.triangle_list[i][1];
                med_tri = sys.triangle_list[i][2];
                low_tri = sys.triangle_list[i][0];
                // sys.my_cout << "low high med" << endl;
            }

            // Note have to do
            // low -> med, low -> high, med -> low, high -> low, med -> high, high -> med
            // 
            // This case is low -> med, low -> high and reverse cases
            // sys.my_cout << "Now onto the sorting hat" << endl;
            if(sys.point_neighbor_list[low_tri][0] == -1){
                // Generalize as triangle list second value is not necessarily 1,2
                // sys.my_cout << "No neighbors" << endl;
                sys.point_neighbor_list[low_tri][0] = med_tri;
                sys.point_neighbor_triangle[low_tri][0][0] = i;
                point_neighbor_list_m[low_tri] += 2;
                // sys.my_cout << "Neighbor list at " << low_tri << " is now " << point_neighbor_list[low_tri][0] << endl;
                // Check opposite direction
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                        sys.point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        // sys.my_cout << "Neighbor list at " << med_tri << " entry " << place_holder_nl <<  " is now " << point_neighbor_list[med_tri][place_holder_nl] << endl;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[med_tri] += 1;
    
                sys.point_neighbor_list[low_tri][1] = high_tri;
                sys.point_neighbor_triangle[low_tri][1][0] = i;
                // sys.my_cout << "Neighbor list at " << low_tri << " is now " << point_neighbor_list[low_tri][1] << endl;
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                        sys.point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        // sys.my_cout << "Neighbor list at " << high_tri << " entry " << place_holder_nl <<  " is now " << point_neighbor_list[high_tri][place_holder_nl] << endl;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[high_tri] += 1;
            }
            // Make big else loop in other case 
            else {
                // sys.my_cout << "Are neighbors initially" << endl;
                // Check to see if low_tri and med_tri are linked
                int link_low_med = LinkTriangleTest(sys, low_tri, med_tri);
                // Convert statements to else as there is no else if if it is not not -1
                if(link_low_med != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[low_tri][place_holder_nl] = med_tri;
                            sys.point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                            sys.point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[med_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_list_m[low_tri]; j++) {
                        if(sys.point_neighbor_list[low_tri][j] == med_tri) {
                            sys.point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_list_m[med_tri]; j++) {
                        if(sys.point_neighbor_list[med_tri][j] == low_tri) {
                            sys.point_neighbor_triangle[med_tri][j][1] = i;
                            break;
                        }
                    }
                }

                int link_low_high = LinkTriangleTest(sys, low_tri, high_tri);
                if(link_low_high != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[low_tri][place_holder_nl] = high_tri;
                            sys.point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < (sys.neighbor_max-1)) {
                        if((sys.point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                            sys.point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                            sys.point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                            place_holder_nl = (sys.neighbor_max-1);
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_list_m[high_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_list_m[low_tri]; j++) {
                        if(sys.point_neighbor_list[low_tri][j] == high_tri) {
                            sys.point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_list_m[high_tri]; j++) {
                        if(sys.point_neighbor_list[high_tri][j] == low_tri) {
                            sys.point_neighbor_triangle[high_tri][j][1] = i;
                            break;
                        }
                    }
                }

            }
            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
                
            // Can use these values to
            // This case is med -> high and reverse cases
            int link_med_high = LinkTriangleTest(sys, med_tri, high_tri);
            if(link_med_high != 1) {
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[med_tri][place_holder_nl] = high_tri;
                        sys.point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[med_tri] += 1;
                place_holder_nl = 0;
                while(place_holder_nl < (sys.neighbor_max-1)) {
                    if((sys.point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        sys.point_neighbor_list[high_tri][place_holder_nl] = med_tri;
                        sys.point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        place_holder_nl = (sys.neighbor_max-1);
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_list_m[high_tri] += 1;
            }
            // If else, then we've seen this link before
            // Add to second point_neighbor_triangle entry
            else {
                for(int j=0; j<point_neighbor_list_m[med_tri]; j++) {
                    if(sys.point_neighbor_list[med_tri][j] == high_tri) {
                        sys.point_neighbor_triangle[med_tri][j][1] = i;
                        break;
                    }
                }
                for(int j=0; j<point_neighbor_list_m[high_tri]; j++) {
                    if(sys.point_neighbor_list[high_tri][j] == med_tri) {
                        sys.point_neighbor_triangle[high_tri][j][1] = i;
                        break;
                    }
                }
            }

            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
            getline(input,line);
        }
    }

    // Now resize all vector entries
    for(int i=0; i<sys.vertices; i++) {
        for(int j=0; j<sys.neighbor_max; j++) {
            if(sys.point_neighbor_list[i][j] == -1) {
                sys.point_neighbor_list[i].resize(j);
                sys.point_neighbor_triangle[i].resize(j);
                break;
            }
            if(sys.point_triangle_list[i][j] == -1) {
                sys.point_triangle_list[i].resize(j);
            }
        }
        for(int j=0; j<sys.neighbor_max; j++) {
            if(sys.point_triangle_list[i][j] == -1) {
                sys.point_triangle_list[i].resize(j);
                break;
            }
        }
    }
    // Set reference original values equal to current entries
    sys.triangle_list_original = sys.triangle_list;
    sys.point_neighbor_list_original = sys.point_neighbor_list;
    sys.point_triangle_list_original = sys.point_triangle_list;
    sys.point_neighbor_triangle_original = sys.point_neighbor_triangle;
}
