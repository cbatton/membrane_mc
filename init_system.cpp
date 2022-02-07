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
#include "saruprng.hpp"
using namespace std;

void InitSystem::InitializeState() {
    for(int i=0; i<dim_x; i++){
        for(int j=0; j<dim_y; j++){
            Radius_x[i][j] = double(Length_x*(i+0.5)/dim_x)/Length_x;
            Radius_y[i][j] = double(Length_y*(j+0.5)/dim_y)/Length_y;
            Radius_z[i][j] = 0.0;
            Radius_x_original[i][j] = Radius_x[i][j];
            Radius_y_original[i][j] = Radius_y[i][j];
            Radius_z_original[i][j] = Radius_z[i][j];
        }
    }
    
    for(int i=0; i<(vertices*num_frac); i++) {
		int j = generator.rand_select(vertices-1);
		while (Ising_Array[j] == 1) {
			j = generator.rand_select(vertices-1);
        }
        Ising_Array[j] = 1;
    }
}

void InitSystem::InitializeEquilState() {
    // Create two layers of points that will triangulate to form a mesh of equilateral triangle
    // Layer one
    for(int i=0; i<dim_x; i += 2){
        for(int j=0; j<dim_y; j++) {
            Radius_x[i][j] = double(Length_x*(i+0.5)/dim_x)/Length_x;
            Radius_y[i][j] = double(Length_y*(j+0.25)/dim_y)/Length_y;
            Radius_z[i][j] = 0.0;
            Radius_x_original[i][j] = Radius_x[i][j];
            Radius_y_original[i][j] = Radius_y[i][j];
            Radius_z_original[i][j] = Radius_z[i][j];
        }
    }
    for(int i=1; i<dim_x; i += 2){
        for(int j=0; j<dim_y; j++) {
            Radius_x[i][j] = double(Length_x*(i+0.5)/dim_x)/Length_x;
            Radius_y[i][j] = double(Length_y*(j+0.75)/dim_y)/Length_y;
            Radius_z[i][j] = 0.0;
            Radius_x_original[i][j] = Radius_x[i][j];
            Radius_y_original[i][j] = Radius_y[i][j];
            Radius_z_original[i][j] = Radius_z[i][j];
        }
    }
    
    for(int i=0; i<(vertices*num_frac); i++) {
		int j = generator.rand_select(vertices-1);
		while (Ising_Array[j] == 1) {
			j = generator.rand_select(vertices-1);
        }
        Ising_Array[j] = 1;
    }    
}

void InitSystem::InputState() {
    ifstream input;
    input.open(output_path+"/config");
    // Check to see if config present. If not, do nothing
    if (input.fail()) {
        cout << "No input configuration file." << endl;
    }

    else{

        char next;
        string line;

        double buffer1 = 0;
		/*
        for(int i=0; i<dim_x;i++){
            for(int j=0; j<dim_y; j++){
                input >> buffer1 >> buffer1 >> buffer1 >> Radius_z[j][i];
            }
        }
		*/
    }

}

void InitSystem::SaruSeed(unsigned int value) {
// Prime Saru with input seeds of seed_base, value, and OpenMP threads
    generator = Saru(seed_base, value);
    #pragma omp parallel for
    for(int i=0; i<omp_get_max_threads(); i++) {
        generators[i] = Saru(seed_base, value, i);
    }
}

void InitSystem::GenerateTriangulation() {
	ofstream myfile;
	myfile.open (output_path+"/out.off");
    cout.rdbuf(myfilebuf);
	myfile << 0 << " " << 0 << " " << Length_x << " " << Length_y << endl;
    myfile << 1 << " " << 1 << endl;
    myfile << vertices << endl;
    for(int i=0; i<dim_x; i++){
		for(int j=0; j<dim_y; j++){	
			myfile << Length_x*Radius_x[i][j] << " " << Length_y*Radius_y[i][j] << endl;
		}
	}
	myfile << endl;
	myfile << faces << endl;
	int triangle_list_active[faces][3];
	int face_count = 0;
    // Add triangles in body
    for(int i=0; i<dim_x-1; i++){
		for(int j=0; j<dim_y-1; j++){
			int dummy_up = j+1;
			int dummy_right = i+1;
            int triangle_trials[2][3];
			triangle_trials[0][0] = j+i*dim_y; triangle_trials[0][1] = j+dummy_right*dim_y; triangle_trials[0][2] = dummy_up+dummy_right*dim_y;
			triangle_trials[1][0] = j+i*dim_y; triangle_trials[1][1] = dummy_up+i*dim_y; triangle_trials[1][2] = dummy_up+dummy_right*dim_y;
            for(int k=0; k<2; k++) {
				triangle_list_active[face_count][0] = triangle_trials[k][0];
				triangle_list_active[face_count][1] = triangle_trials[k][1];
				triangle_list_active[face_count][2] = triangle_trials[k][2];
				face_count++;
				myfile << triangle_trials[k][0] << " " << triangle_trials[k][1] << " " << triangle_trials[k][2] << endl;
			}
		}		
	}
    // cout << face_count << endl;
	myfile.close();
}

void InitSystem::GenerateTriangulationEquil() {
	ofstream myfile;
	myfile.open (output_path+"/out.off");
    cout.rdbuf(myfilebuf);
	myfile << 0 << " " << 0 << " " << Length_x << " " << Length_y << endl;
    myfile << 1 << " " << 1 << endl;
    myfile << vertices << endl;
    for(int i=0; i<dim_x; i++){
		for(int j=0; j<dim_y; j++){	
			myfile << Length_x*Radius_x[i][j] << " " << Length_y*Radius_y[i][j] << endl;
		}
	}
	myfile << endl;
	myfile << faces << endl;
	int triangle_list_active[faces][3];
	int face_count = 0;
    // Add triangles in body
    for(int i=0; i<dim_x; i++){
        for(int j=0; j<dim_y; j++){
            if(i%2==0) {
				int dummy_up = j+1;
				if(dummy_up >= dim_y) {
					dummy_up -= dim_y;
				}
				int dummy_down = j-1;
				if(dummy_down < 0) {
					dummy_down += dim_y;
				}
				int dummy_right = i+1;
				if(dummy_right >= dim_x) {
					dummy_right -= dim_x;
				}
				int dummy_left = i-1;
				if(dummy_left < 0) {
					dummy_left += dim_x;
				}
                // Basic idea here is to form the six triangles around each point, and then check to see if they are on the active triangle list
                // If not, add to active triangle list and write to myfile
                // New idea: generate six neighboring points, attempt to add to 
                int point_neighbor_trials[6];
                point_neighbor_trials[0] = dummy_up+i*dim_y;
                point_neighbor_trials[1] = j+dummy_right*dim_y;
                point_neighbor_trials[2] = dummy_down+dummy_right*dim_y;
                point_neighbor_trials[3] = dummy_down+i*dim_y;
                point_neighbor_trials[4] = j+dummy_left*dim_y;
                point_neighbor_trials[5] = dummy_down+dummy_left*dim_y;
                int triangle_trials[6][3];
                triangle_trials[0][0] = j+i*dim_y; triangle_trials[0][1] = dummy_up+i*dim_y; triangle_trials[0][2] = j+dummy_right*dim_y;
                triangle_trials[1][0] = j+i*dim_y; triangle_trials[1][1] = dummy_down+dummy_right*dim_y; triangle_trials[1][2] = j+dummy_right*dim_y;
                triangle_trials[2][0] = j+i*dim_y; triangle_trials[2][1] = dummy_down+i*dim_y; triangle_trials[2][2] = dummy_down+dummy_right*dim_y;
                triangle_trials[3][0] = j+i*dim_y; triangle_trials[3][1] = dummy_up+i*dim_y; triangle_trials[3][2] = j+dummy_left*dim_y;
                triangle_trials[4][0] = j+i*dim_y; triangle_trials[4][1] = dummy_down+dummy_left*dim_y; triangle_trials[4][2] = j+dummy_left*dim_y;
                triangle_trials[5][0] = j+i*dim_y; triangle_trials[5][1] = dummy_down+i*dim_y; triangle_trials[5][2] = dummy_down+dummy_left*dim_y;
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
				if(dummy_up >= dim_y) {
					dummy_up -= dim_y;
				}
				int dummy_down = j-1;
				if(dummy_down < 0) {
					dummy_down += dim_y;
				}
				int dummy_right = i+1;
				if(dummy_right >= dim_x) {
					dummy_right -= dim_x;
				}
				int dummy_left = i-1;
				if(dummy_left < 0) {
					dummy_left += dim_x;
				}
                // Basic idea here is to form the six triangles around each point, and then check to see if they are on the active triangle list
                // If not, add to active triangle list and write to myfile
                // New idea: generate six neighboring points, attempt to add to 
                int point_neighbor_trials[6];
                point_neighbor_trials[0] = dummy_up+i*dim_y;
                point_neighbor_trials[1] = dummy_up+dummy_right*dim_y;
                point_neighbor_trials[2] = j+dummy_right*dim_y;
                point_neighbor_trials[3] = dummy_down+i*dim_y;
                point_neighbor_trials[4] = dummy_up+dummy_left*dim_y;
                point_neighbor_trials[5] = j+dummy_left*dim_y;
                int triangle_trials[6][3];
                triangle_trials[0][0] = j+i*dim_y; triangle_trials[0][1] = dummy_up+i*dim_y; triangle_trials[0][2] = dummy_up+dummy_right*dim_y;
                triangle_trials[1][0] = j+i*dim_y; triangle_trials[1][1] = dummy_up+dummy_right*dim_y; triangle_trials[1][2] = j+dummy_right*dim_y;
                triangle_trials[2][0] = j+i*dim_y; triangle_trials[2][1] = dummy_down+i*dim_y; triangle_trials[2][2] = j+dummy_right*dim_y;
                triangle_trials[3][0] = j+i*dim_y; triangle_trials[3][1] = dummy_up+i*dim_y; triangle_trials[3][2] = dummy_up+dummy_left*dim_y;
                triangle_trials[4][0] = j+i*dim_y; triangle_trials[4][1] = dummy_up+dummy_left*dim_y; triangle_trials[4][2] = j+dummy_left*dim_y;
                triangle_trials[5][0] = j+i*dim_y; triangle_trials[5][1] = dummy_down+i*dim_y; triangle_trials[5][2] = j+dummy_left*dim_y;
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
    // cout << face_count << endl;
	myfile.close();
}

inline int InitSystem::LinkTriangleTest(int vertex_1, int vertex_2) {
// Determines if two points are linked together
    int link_1_2 = 0;
    for(int i=0; i<point_neighbor_max[vertex_1]; i++) {
        if(point_neighbor_list[vertex_1][i] == vertex_2) {
            link_1_2 += 1;
            break;
        }
    }
    return link_1_2;
}

inline void InitSystem::LinkTriangleFace(int vertex_1, int vertex_2, int face[2]) {
// Determines which triangles two linked points are mutually on
    int face_1 = -1;
    int face_2 = -1;    
    for(int i=0; i<point_triangle_max[vertex_1]; i++) {
        for(int j=0; j<point_triangle_max[vertex_2]; j++) {
            if(point_triangle_list[vertex_1][i] == point_triangle_list[vertex_2][j]) {
                if(face_1 == -1) {
                    face_1 = point_triangle_list[vertex_1][i];
                }
                else {
                    face_2 = point_triangle_list[vertex_1][i];
                    i += point_triangle_max[vertex_1];
                    j += point_triangle_max[vertex_2];
                }                
            }
        }
    }
    face[0] = face_1;
    face[1] = face_2;
}

void InitSystem::UseTriangulation(string name) {
    ifstream input;
    input.open(output_path+"/"+name);
    cout.rdbuf(myfilebuf);
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        cout << "No input file." << endl;
    }

    else{
        char buffer;
        string line;
        // cout << "Connectivity file detected. Changing values." << endl;
        // Skip 3 lines
        for(int i=0; i<3; i++){
            getline(input,line);
        }
        // Input radius values
        
		double scale_factor = pow(scale,0.5);
        for(int i=0; i<active_vertices; i++){
            input >> Radius_x_tri[i] >> Radius_y_tri[i];
            Radius_z_tri[i] = 0;
			// Rescale coordinates
			/*
			Radius_x_tri[i] *= scale_factor;
			Radius_y_tri[i] *= scale_factor;
			Radius_z_tri[i] *= scale_factor;
			*/
            Radius_x_tri[i] = Radius_x_tri[i]/Length_x;
            Radius_y_tri[i] = Radius_y_tri[i]/Length_y;
			// cout << "x y z " << Length_x*Radius_x_tri[i] << " " << Length_y*Radius_y_tri[i] << " " << Radius_z_tri[i] << endl;
            Radius_x_tri_original[i] = Radius_x_tri[i];
            Radius_y_tri_original[i] = Radius_y_tri[i];
			Radius_z_tri_original[i] = Radius_z_tri[i];
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
        // In while loop skipping negative numbers, if to be added value is unique add to first entry of link_triangle_list
        // In while loop skipping negative numbers, if to be added value is the same as already in place value add to second entry of link_triangle_list
        // link_triangle_list is needed to compute needed cotan angles by evaluating angle through standard formula with other links in triangle
        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++){
                point_neighbor_list[i][j] = -1;
                point_neighbor_triangle[i][j][0] = -1;
                point_neighbor_triangle[i][j][1] = -1;
            }
        }
        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++) {
                point_triangle_list[i][j] = -1;
            }
        }

        for(int i=0; i<active_vertices; i++) {
            point_neighbor_max[i] = 0;
            point_triangle_max[i] = 0;
        }

        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++){
                point_neighbor_list_original[i][j] = -1;
                point_neighbor_triangle_original[i][j][0] = -1;
                point_neighbor_triangle_original[i][j][1] = -1;
            }
        }
        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++) {
                point_triangle_list_original[i][j] = -1;
            }
        }

        for(int i=0; i<active_vertices; i++) {
            point_neighbor_max_original[i] = 0;
            point_triangle_max_original[i] = 0;
        }

        for(int i=0; i<active_faces; i++){
            input >> triangle_list[i][0] >> triangle_list[i][1] >> triangle_list[i][2];
            // Keep orientation consistent here
            // Initial basis is all normals pointing up in z-direction
            double normal_test[3] = {0,0,0};
            normalTriangle(i, normal_test);
            if(normal_test[2] < 0) {
                // If so, swapping first two entries will make normals okay
                int triangle_list_0 = triangle_list[i][0];
                triangle_list[i][0] = triangle_list[i][1];
                triangle_list[i][1] = triangle_list_0;
            }
            // cout << "For face " << i << " the points are " << triangle_list[i][0] << " " << triangle_list[i][1] << " " << triangle_list[i][2] << endl;
            // Most likely don't need this for loop, just need to add connections from 0->1, 1->2, 2->1
            int place_holder_nl = 0;

            // Add to point_triangle_list
            if (point_triangle_list[triangle_list[i][0]][0] == -1) {
                point_triangle_list[triangle_list[i][0]][0] = i;
                point_triangle_max[triangle_list[i][0]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < neighbor_max) {
                    if(point_triangle_list[triangle_list[i][0]][place_holder_nl] == -1) {
                        point_triangle_list[triangle_list[i][0]][place_holder_nl] = i;
                        place_holder_nl = neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_max[triangle_list[i][0]] += 1;
            }
            if (point_triangle_list[triangle_list[i][1]][0] == -1) {
                point_triangle_list[triangle_list[i][1]][0] = i;
                point_triangle_max[triangle_list[i][1]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < neighbor_max) {
                    if(point_triangle_list[triangle_list[i][1]][place_holder_nl] == -1) {
                        point_triangle_list[triangle_list[i][1]][place_holder_nl] = i;
                        place_holder_nl = neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_max[triangle_list[i][1]] += 1;
            }
            if (point_triangle_list[triangle_list[i][2]][0] == -1) {
                point_triangle_list[triangle_list[i][2]][0] = i;
                point_triangle_max[triangle_list[i][2]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < neighbor_max) {
                    if(point_triangle_list[triangle_list[i][2]][place_holder_nl] == -1) {
                        point_triangle_list[triangle_list[i][2]][place_holder_nl] = i;
                        place_holder_nl = neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_max[triangle_list[i][2]] += 1;
            }
            // For loop variable was j. Now, idea is the following
            // Sort entries of triangle_list so we have low, medium, high
            // Do link list as doing that by [lower][other] on the vertex, so do that for low -> medium, low -> high, medium -> high
            // Thus go in order of low (most steps), medium (less steps), high (just add to point_neighbor_list)
            int low_tri;
            int med_tri;
            int high_tri;
            if ((triangle_list[i][0] > triangle_list[i][1]) && (triangle_list[i][1] > triangle_list[i][2])) {
                high_tri = triangle_list[i][0];
                med_tri = triangle_list[i][1];
                low_tri = triangle_list[i][2];
                // cout << "high med low" << endl;
            }
            else if((triangle_list[i][1] > triangle_list[i][0]) && (triangle_list[i][0] > triangle_list[i][2])) {
                high_tri = triangle_list[i][1];
                med_tri = triangle_list[i][0];
                low_tri = triangle_list[i][2];
                // cout << "med high low" << endl;
            }
            else if((triangle_list[i][2] > triangle_list[i][0]) && (triangle_list[i][0] > triangle_list[i][1])) {
                high_tri = triangle_list[i][2];
                med_tri = triangle_list[i][0];
                low_tri = triangle_list[i][1];
                // cout << "med low high" << endl;
            }
            else if((triangle_list[i][2] > triangle_list[i][1]) && (triangle_list[i][1] > triangle_list[i][0])) {
                high_tri = triangle_list[i][2];
                med_tri = triangle_list[i][1];
                low_tri = triangle_list[i][0];
                // cout << "low med high" << endl;
            }
            else if((triangle_list[i][0] > triangle_list[i][2]) && (triangle_list[i][2] > triangle_list[i][1])) {
                high_tri = triangle_list[i][0];
                med_tri = triangle_list[i][2];
                low_tri = triangle_list[i][1];
                // cout << "high low med" << endl;
            }
            else {
                high_tri = triangle_list[i][1];
                med_tri = triangle_list[i][2];
                low_tri = triangle_list[i][0];
                // cout << "low high med" << endl;
            }

            // Note have to do
            // low -> med, low -> high, med -> low, high -> low, med -> high, high -> med
            // 
            // This case is low -> med, low -> high and reverse cases
            // cout << "Now onto the sorting hat" << endl;
            if(point_neighbor_list[low_tri][0] == -1){
                // Generalize as triangle list second value is not necessarily 1,2
                // cout << "No neighbors" << endl;
                point_neighbor_list[low_tri][0] = med_tri;
                point_neighbor_triangle[low_tri][0][0] = i;
                point_neighbor_max[low_tri] += 2;
                // cout << "Neighbor list at " << low_tri << " is now " << point_neighbor_list[low_tri][0] << endl;
                // Add to link list statement, if else statement depending on relative values, adding to first entry
                // link_triangle_list[low_tri][med_tri][0] = i;
                // cout << "Link list at " << low_tri << " " << med_tri << " is now " << link_triangle_list[low_tri][med_tri][0] << endl;
                // Check opposite direction
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                        point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        // cout << "Neighbor list at " << med_tri << " entry " << place_holder_nl <<  " is now " << point_neighbor_list[med_tri][place_holder_nl] << endl;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[med_tri] += 1;
    
                point_neighbor_list[low_tri][1] = high_tri;
                point_neighbor_triangle[low_tri][1][0] = i;
                // cout << "Neighbor list at " << low_tri << " is now " << point_neighbor_list[low_tri][1] << endl;
                // Add to link list statement in first entry
                // link_triangle_list[low_tri][high_tri][0] = i;
                // cout << "Link list at " << low_tri << " " << high_tri << " is now " << link_triangle_list[low_tri][high_tri][0] << endl;
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                        point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        // cout << "Neighbor list at " << high_tri << " entry " << place_holder_nl <<  " is now " << point_neighbor_list[high_tri][place_holder_nl] << endl;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[high_tri] += 1;
            }
            // Make big else loop in other case 
            // Can use link lists to skip iteration and jump to switching second link list value
            // Then if that is not true, iterate
            // Note that if link_triangle_list first entry is not -1, then it has already been visited so we can skip
            else {
                // cout << "Are neighbors initially" << endl;
                // Check to see if low_tri and med_tri are linked
                int link_low_med = link_triangle_test(low_tri, med_tri);
                // Convert statements to else as there is no else if if it is not not -1
                if(link_low_med != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[low_tri][place_holder_nl] = med_tri;
                            point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                            point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[med_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_max[low_tri]; j++) {
                        if(point_neighbor_list[low_tri][j] == med_tri) {
                            point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_max[med_tri]; j++) {
                        if(point_neighbor_list[med_tri][j] == low_tri) {
                            point_neighbor_triangle[med_tri][j][1] = i;
                            break;
                        }
                    }
                }

                int link_low_high = link_triangle_test(low_tri, high_tri);
                if(link_low_high != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[low_tri][place_holder_nl] = high_tri;
                            point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                            point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[high_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_max[low_tri]; j++) {
                        if(point_neighbor_list[low_tri][j] == high_tri) {
                            point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_max[high_tri]; j++) {
                        if(point_neighbor_list[high_tri][j] == low_tri) {
                            point_neighbor_triangle[high_tri][j][1] = i;
                            break;
                        }
                    }
                }

            }
            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
                
            // Can use these values to
            // This case is med -> high and reverse cases
            int link_med_high = link_triangle_test(med_tri, high_tri);
            if(link_med_high != 1) {
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[med_tri][place_holder_nl] = high_tri;
                        point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[med_tri] += 1;
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[high_tri][place_holder_nl] = med_tri;
                        point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[high_tri] += 1;
            }
            // If else, then we've seen this link before
            // Add to second point_neighbor_triangle entry
            else {
                for(int j=0; j<point_neighbor_max[med_tri]; j++) {
                    if(point_neighbor_list[med_tri][j] == high_tri) {
                        point_neighbor_triangle[med_tri][j][1] = i;
                        break;
                    }
                }
                for(int j=0; j<point_neighbor_max[high_tri]; j++) {
                    if(point_neighbor_list[high_tri][j] == med_tri) {
                        point_neighbor_triangle[high_tri][j][1] = i;
                        break;
                    }
                }
            }

            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
            getline(input,line);
        }
        // int point_neighbor_list[vertices][10];
        // int link_triangle_list[vertices][vertices][2];

    }

    // Set reference original values equal to current entries
    for(int i=0; i<faces; i++) {
        for(int j=0; j<3; j++) {
            triangle_list_original[i][j] = triangle_list[i][j];
        }
    }

    for(int i=0; i<vertices; i++) {
        for(int j=0; j<neighbor_max; j++) {
            point_neighbor_list_original[i][j] = point_neighbor_list[i][j];
            point_triangle_list_original[i][j] = point_triangle_list[i][j];
            for(int k=0; k<2; k++) {
                point_neighbor_triangle_original[i][j][k] = point_neighbor_triangle[i][j][k];
            }
        }
        point_neighbor_max_original[i] = point_neighbor_max[i];
        point_triangle_max_original[i] = point_triangle_max[i];
    }

    for(int i=0; i<active_vertices; i++) {
        // cout << "Max neighbors at " << i << " is " << point_neighbor_max[i] << endl;
        // cout << "Max triangles at " << i << " is " << point_triangle_max[i] << endl;
    }

    /*
    for(int i=0; i<active_vertices; i++) {
        for(int j=0; j<active_vertices; j++) { 
            cout << "Link list at " << i << " " << j << " is 1: " << link_triangle_list[i][j][0] << " and 2: " << link_triangle_list[i][j][1] << endl;
        }
    }

    for(int i=0; i<active_vertices; i++) {
        for(int j=0; j<10; j++){
            cout << "Neighbor list at " << i << " entry " << j << " :" << point_neighbor_list[i][j] << endl;
        }
    }

    for(int i=0; i<active_vertices; i++) {
        cout << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<10; j++) {
            cout << point_triangle_list[i][j] << " ";
        }
        cout << endl;
    }
    */

}

void InitSystem::UseTriangulationRestart(string name) {
    ifstream input;
    input.open(output_path+"/"+name);
    cout.rdbuf(myfilebuf);
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        cout << "No input file." << endl;
    }

    else{
        char buffer;
        string line;
        // cout << "Connectivity file detected. Changing values." << endl;
        // Get box size
        input >> line >> Length_x >> line >> Length_y >> line >> Length_z >> line >> count_step;
        scale_xy = Length_x/Length_x_base;
        Length_x_old = Length_x;
        Length_y_old = Length_y;
        Length_z_old = Length_z;
        // Skip 2 lines
        for(int i=0; i<2; i++){
            getline(input,line);
        } 
        // Input radius values
		double scale_factor = pow(scale,0.5);
        for(int i=0; i<active_vertices; i++){
            input >> Ising_Array[i] >> Radius_x_tri[i] >> Radius_y_tri[i] >> Radius_z_tri[i];
            if(Ising_Array[i] == 2) {
                protein_node[i] = 0;
            }
			// Rescale coordinates
			/*
			Radius_x_tri[i] *= scale_factor;
			Radius_y_tri[i] *= scale_factor;
			Radius_z_tri[i] *= scale_factor;
			*/
            Radius_x_tri[i] = Radius_x_tri[i]/Length_x;
            Radius_y_tri[i] = Radius_y_tri[i]/Length_y;
			// cout << "x y z " << Length_x*Radius_x_tri[i] << " " << Length_y*Radius_y_tri[i] << " " << Radius_z_tri[i] << endl;
            Radius_x_tri_original[i] = Radius_x_tri[i];
            Radius_y_tri_original[i] = Radius_y_tri[i];
			Radius_z_tri_original[i] = Radius_z_tri[i];
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
        // In while loop skipping negative numbers, if to be added value is unique add to first entry of link_triangle_list
        // In while loop skipping negative numbers, if to be added value is the same as already in place value add to second entry of link_triangle_list
        // link_triangle_list is needed to compute needed cotan angles by evaluating angle through standard formula with other links in triangle
        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++){
                point_neighbor_list[i][j] = -1;
                point_neighbor_triangle[i][j][0] = -1;
                point_neighbor_triangle[i][j][1] = -1;
            }
        }
        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++) {
                point_triangle_list[i][j] = -1;
            }
        }

        for(int i=0; i<active_vertices; i++) {
            point_neighbor_max[i] = 0;
            point_triangle_max[i] = 0;
        }

        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++){
                point_neighbor_list_original[i][j] = -1;
                point_neighbor_triangle_original[i][j][0] = -1;
                point_neighbor_triangle_original[i][j][1] = -1;
            }
        }
        for(int i=0; i<active_vertices; i++) {
            for(int j=0; j<neighbor_max; j++) {
                point_triangle_list_original[i][j] = -1;
            }
        }

        for(int i=0; i<active_vertices; i++) {
            point_neighbor_max_original[i] = 0;
            point_triangle_max_original[i] = 0;
        }
        for(int i=0; i<active_faces; i++){
            input >> triangle_list[i][0] >> triangle_list[i][1] >> triangle_list[i][2];
            // cout << "For face " << i << " the points are " << triangle_list[i][0] << " " << triangle_list[i][1] << " " << triangle_list[i][2] << endl;
            // Most likely don't need this for loop, just need to add connections from 0->1, 1->2, 2->1
            int place_holder_nl = 0;

            // Add to point_triangle_list
            if (point_triangle_list[triangle_list[i][0]][0] == -1) {
                point_triangle_list[triangle_list[i][0]][0] = i;
                point_triangle_max[triangle_list[i][0]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < neighbor_max) {
                    if(point_triangle_list[triangle_list[i][0]][place_holder_nl] == -1) {
                        point_triangle_list[triangle_list[i][0]][place_holder_nl] = i;
                        place_holder_nl = neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_max[triangle_list[i][0]] += 1;
            }
            if (point_triangle_list[triangle_list[i][1]][0] == -1) {
                point_triangle_list[triangle_list[i][1]][0] = i;
                point_triangle_max[triangle_list[i][1]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < neighbor_max) {
                    if(point_triangle_list[triangle_list[i][1]][place_holder_nl] == -1) {
                        point_triangle_list[triangle_list[i][1]][place_holder_nl] = i;
                        place_holder_nl = neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_max[triangle_list[i][1]] += 1;
            }
            if (point_triangle_list[triangle_list[i][2]][0] == -1) {
                point_triangle_list[triangle_list[i][2]][0] = i;
                point_triangle_max[triangle_list[i][2]] += 1;
            }
            else {
                place_holder_nl = 1;
                while(place_holder_nl < neighbor_max) {
                    if(point_triangle_list[triangle_list[i][2]][place_holder_nl] == -1) {
                        point_triangle_list[triangle_list[i][2]][place_holder_nl] = i;
                        place_holder_nl = neighbor_max;
                    }
                    place_holder_nl += 1;
                }
                place_holder_nl = 0;
                point_triangle_max[triangle_list[i][2]] += 1;
            }
            // For loop variable was j. Now, idea is the following
            // Sort entries of triangle_list so we have low, medium, high
            // Do link list as doing that by [lower][other] on the vertex, so do that for low -> medium, low -> high, medium -> high
            // Thus go in order of low (most steps), medium (less steps), high (just add to point_neighbor_list)
            int low_tri;
            int med_tri;
            int high_tri;
            if ((triangle_list[i][0] > triangle_list[i][1]) && (triangle_list[i][1] > triangle_list[i][2])) {
                high_tri = triangle_list[i][0];
                med_tri = triangle_list[i][1];
                low_tri = triangle_list[i][2];
                // cout << "high med low" << endl;
            }
            else if((triangle_list[i][1] > triangle_list[i][0]) && (triangle_list[i][0] > triangle_list[i][2])) {
                high_tri = triangle_list[i][1];
                med_tri = triangle_list[i][0];
                low_tri = triangle_list[i][2];
                // cout << "med high low" << endl;
            }
            else if((triangle_list[i][2] > triangle_list[i][0]) && (triangle_list[i][0] > triangle_list[i][1])) {
                high_tri = triangle_list[i][2];
                med_tri = triangle_list[i][0];
                low_tri = triangle_list[i][1];
                // cout << "med low high" << endl;
            }
            else if((triangle_list[i][2] > triangle_list[i][1]) && (triangle_list[i][1] > triangle_list[i][0])) {
                high_tri = triangle_list[i][2];
                med_tri = triangle_list[i][1];
                low_tri = triangle_list[i][0];
                // cout << "low med high" << endl;
            }
            else if((triangle_list[i][0] > triangle_list[i][2]) && (triangle_list[i][2] > triangle_list[i][1])) {
                high_tri = triangle_list[i][0];
                med_tri = triangle_list[i][2];
                low_tri = triangle_list[i][1];
                // cout << "high low med" << endl;
            }
            else {
                high_tri = triangle_list[i][1];
                med_tri = triangle_list[i][2];
                low_tri = triangle_list[i][0];
                // cout << "low high med" << endl;
            }

            // Note have to do
            // low -> med, low -> high, med -> low, high -> low, med -> high, high -> med
            // 
            // This case is low -> med, low -> high and reverse cases
            // cout << "Now onto the sorting hat" << endl;
            if(point_neighbor_list[low_tri][0] == -1){
                // Generalize as triangle list second value is not necessarily 1,2
                // cout << "No neighbors" << endl;
                point_neighbor_list[low_tri][0] = med_tri;
                point_neighbor_triangle[low_tri][0][0] = i;
                point_neighbor_max[low_tri] += 2;
                // cout << "Neighbor list at " << low_tri << " is now " << point_neighbor_list[low_tri][0] << endl;
                // Add to link list statement, if else statement depending on relative values, adding to first entry
                // link_triangle_list[low_tri][med_tri][0] = i;
                // cout << "Link list at " << low_tri << " " << med_tri << " is now " << link_triangle_list[low_tri][med_tri][0] << endl;
                // Check opposite direction
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                        point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        // cout << "Neighbor list at " << med_tri << " entry " << place_holder_nl <<  " is now " << point_neighbor_list[med_tri][place_holder_nl] << endl;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[med_tri] += 1;
    
                point_neighbor_list[low_tri][1] = high_tri;
                point_neighbor_triangle[low_tri][1][0] = i;
                // cout << "Neighbor list at " << low_tri << " is now " << point_neighbor_list[low_tri][1] << endl;
                // Add to link list statement in first entry
                // link_triangle_list[low_tri][high_tri][0] = i;
                // cout << "Link list at " << low_tri << " " << high_tri << " is now " << link_triangle_list[low_tri][high_tri][0] << endl;
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                        point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        // cout << "Neighbor list at " << high_tri << " entry " << place_holder_nl <<  " is now " << point_neighbor_list[high_tri][place_holder_nl] << endl;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[high_tri] += 1;
            }
            // Make big else loop in other case 
            // Can use link lists to skip iteration and jump to switching second link list value
            // Then if that is not true, iterate
            // Note that if link_triangle_list first entry is not -1, then it has already been visited so we can skip
            else {
                // cout << "Are neighbors initially" << endl;
                // Check to see if low_tri and med_tri are linked
                int link_low_med = link_triangle_test(low_tri, med_tri);
                // Convert statements to else as there is no else if if it is not not -1
                if(link_low_med != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[low_tri][place_holder_nl] = med_tri;
                            point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[med_tri][place_holder_nl] = low_tri;
                            point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[med_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_max[low_tri]; j++) {
                        if(point_neighbor_list[low_tri][j] == med_tri) {
                            point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_max[med_tri]; j++) {
                        if(point_neighbor_list[med_tri][j] == low_tri) {
                            point_neighbor_triangle[med_tri][j][1] = i;
                            break;
                        }
                    }
                }

                int link_low_high = link_triangle_test(low_tri, high_tri);
                if(link_low_high != 1) {
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[low_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[low_tri][place_holder_nl] = high_tri;
                            point_neighbor_triangle[low_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[low_tri] += 1;
                    place_holder_nl = 0;
                    while(place_holder_nl < 9) {
                        if((point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                            point_neighbor_list[high_tri][place_holder_nl] = low_tri;
                            point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                            place_holder_nl = 9;
                        }
                        place_holder_nl += 1;
                    }
                    point_neighbor_max[high_tri] += 1;
                }
                // If else, then we've seen this link before
                // Add to second point_neighbor_triangle entry
                else {
                    for(int j=0; j<point_neighbor_max[low_tri]; j++) {
                        if(point_neighbor_list[low_tri][j] == high_tri) {
                            point_neighbor_triangle[low_tri][j][1] = i;
                            break;
                        }
                    }
                    for(int j=0; j<point_neighbor_max[high_tri]; j++) {
                        if(point_neighbor_list[high_tri][j] == low_tri) {
                            point_neighbor_triangle[high_tri][j][1] = i;
                            break;
                        }
                    }
                }

            }
            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
                
            // Can use these values to
            // This case is med -> high and reverse cases
            int link_med_high = link_triangle_test(med_tri, high_tri);
            if(link_med_high != 1) {
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[med_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[med_tri][place_holder_nl] = high_tri;
                        point_neighbor_triangle[med_tri][place_holder_nl][0] = i;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[med_tri] += 1;
                place_holder_nl = 0;
                while(place_holder_nl < 9) {
                    if((point_neighbor_list[high_tri][place_holder_nl] == -1)) {
                        point_neighbor_list[high_tri][place_holder_nl] = med_tri;
                        point_neighbor_triangle[high_tri][place_holder_nl][0] = i;
                        place_holder_nl = 9;
                    }
                    place_holder_nl += 1;
                }
                point_neighbor_max[high_tri] += 1;
            }
            // If else, then we've seen this link before
            // Add to second point_neighbor_triangle entry
            else {
                for(int j=0; j<point_neighbor_max[med_tri]; j++) {
                    if(point_neighbor_list[med_tri][j] == high_tri) {
                        point_neighbor_triangle[med_tri][j][1] = i;
                        break;
                    }
                }
                for(int j=0; j<point_neighbor_max[high_tri]; j++) {
                    if(point_neighbor_list[high_tri][j] == med_tri) {
                        point_neighbor_triangle[high_tri][j][1] = i;
                        break;
                    }
                }
            }

            // else statement looping through first positive values to first avaliable negative value
            // If value is equal along the way, add to second entry of link list
            getline(input,line);
        }
        // int point_neighbor_list[vertices][10];
        // int link_triangle_list[vertices][vertices][2];

    }

    // Set reference original values equal to current entries
    for(int i=0; i<faces; i++) {
        for(int j=0; j<3; j++) {
            triangle_list_original[i][j] = triangle_list[i][j];
        }
    }

    for(int i=0; i<vertices; i++) {
        for(int j=0; j<neighbor_max; j++) {
            point_neighbor_list_original[i][j] = point_neighbor_list[i][j];
            point_triangle_list_original[i][j] = point_triangle_list[i][j];
            for(int k=0; k<2; k++) {
                point_neighbor_triangle_original[i][j][k] = point_neighbor_triangle[i][j][k];
            }
        }
        point_neighbor_max_original[i] = point_neighbor_max[i];
        point_triangle_max_original[i] = point_triangle_max[i];
    }

    for(int i=0; i<active_vertices; i++) {
        // cout << "Max neighbors at " << i << " is " << point_neighbor_max[i] << endl;
        // cout << "Max triangles at " << i << " is " << point_triangle_max[i] << endl;
    }

    /*
    for(int i=0; i<active_vertices; i++) {
        for(int j=0; j<active_vertices; j++) { 
            cout << "Link list at " << i << " " << j << " is 1: " << link_triangle_list[i][j][0] << " and 2: " << link_triangle_list[i][j][1] << endl;
        }
    }

    for(int i=0; i<active_vertices; i++) {
        for(int j=0; j<10; j++){
            cout << "Neighbor list at " << i << " entry " << j << " :" << point_neighbor_list[i][j] << endl;
        }
    }

    for(int i=0; i<active_vertices; i++) {
        cout << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<10; j++) {
            cout << point_triangle_list[i][j] << " ";
        }
        cout << endl;
    }
    */
}
