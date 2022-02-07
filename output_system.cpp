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
#include "output_system.hpp"
using namespace std;

void OutputSystem::OutputTriangulation(string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.open(output_path+"/"+name);
    myfile.precision(17);
    myfile << "Length_x " << std::scientific << Length_x << " Length_y " << std::scientific << Length_y << " Length_z " << std::scientific << Length_z << " count_step " << count_step << endl;
    myfile << active_vertices << " " << active_faces << " " << 0 << endl;
	myfile << endl;
	// Input radius values
	for(int i=0; i<active_vertices; i++){
		myfile << Ising_Array[i] << " " << std::scientific << Length_x*Radius_x_tri[i] << " " << std::scientific << Length_y*Radius_y_tri[i] << " " << std::scientific << Radius_z_tri[i] << "\n";
	}

    // Skip 2 lines
    for(int i=0; i<2; i++){
        myfile << "\n";
    }
	// Input triangle connectivity
	for(int i=0; i<active_faces; i++){
		myfile << triangle_list[i][0] << " " << triangle_list[i][1] << " " << triangle_list[i][2] << "\n";
	}
    myfile.close();
}

void OutputSystem::OutputTriangulationAppend(string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.open(output_path+"/"+name, std::ios_base::app);
    myfile << "Length_x " << std::scientific << Length_x << " Length_y " << std::scientific << Length_y << " Length_z " << std::scientific << Length_z << " count_step " << count_step << endl;
    myfile << active_vertices << " " << active_faces << " " << 0 << endl;
	myfile << endl;
	for(int i=0; i<active_vertices; i++){
		myfile << Ising_Array[i] << " " << std::scientific << Length_x*Radius_x_tri[i] << " " << std::scientific << Length_y*Radius_y_tri[i] << " " << std::scientific << Radius_z_tri[i] << "\n";
	}

    // Skip 2 lines
    for(int i=0; i<2; i++){
        myfile << "\n";
    }
	// Input triangle connectivity
	for(int i=0; i<active_faces; i++){
		myfile << triangle_list[i][0] << " " << triangle_list[i][1] << " " << triangle_list[i][2] << "\n";
	}
    myfile.close();
}

void OutputSystem::OutputTriangulationStorage() {
    ofstream myfile;

    myfile.open (output_path+"/triangle_list.txt", std::ios_base::app);
    for(int i=0; i<active_faces; i++){
        myfile << "For face " << i << " the points are " << triangle_list[i][0] << " " << triangle_list[i][1] << " " << triangle_list[i][2] << endl;
    }
    myfile.close();

    myfile.open (output_path+"/max_numbers.txt", std::ios_base::app);
    for(int i=0; i<active_vertices; i++) {
        myfile << "Max neighbors at " << i << " is " << point_neighbor_max[i] << endl;
        myfile << "Max triangles at " << i << " is " << point_triangle_max[i] << endl;
    }
    myfile.close();

    myfile.open (output_path+"/neighbor_list.txt", std::ios_base::app);
    for(int i=0; i<active_vertices; i++) {
        myfile << "Neighbor list at vertex " << i << " is given by : ";
        for(int j=0; j<neighbor_max; j++){
            myfile << point_neighbor_list[i][j] << " ";
        }
        myfile << endl;
    }
    myfile.close();

    myfile.open (output_path+"/point_triangle_list.txt", std::ios_base::app);
    for(int i=0; i<active_vertices; i++) {
        myfile << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<neighbor_max; j++) {
            myfile << point_triangle_list[i][j] << " ";
        }
        myfile << endl;
    }
    myfile.close();

}

void OutputSystem::DumpXYZConfig(string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.precision(6);
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << vertices << endl;
    myfile << "# step " << count_step << " Length_x " << Length_x << endl;
    for(int i=1; i <= vertices; i++) {
        myfile << " " << Ising_Array[i-1] << " " << std::scientific << Length_x*Radius_x_tri[i-1] << " " << std::scientific << Length_y*Radius_y_tri[i-1] << " " << std::scientific << Radius_z_tri[i-1] << "\n";
        // myfile << " " << Ising_Array[i-1] << " " << std::scientific << Radius_x_tri[i-1] << " " << std::scientific << Radius_y_tri[i-1] << " " << std::scientific << Radius_z_tri[i-1] << endl;
    }
    myfile.close();
}

void OutputSystem::DumpXYZConfigNormal(string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.precision(6);
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << vertices+2.0*faces << endl;
    myfile << "# step " << count_step << " Length_x " << Length_x << endl;
    for(int i=1; i <= vertices; i++) {
        myfile << " " << Ising_Array[i-1] << " " << std::scientific << Length_x*Radius_x_tri[i-1] << " " << std::scientific << Length_y*Radius_y_tri[i-1] << " " << std::scientific << Radius_z_tri[i-1] << "\n";
        // myfile << " " << Ising_Array[i-1] << " " << std::scientific << Radius_x_tri[i-1] << " " << std::scientific << Radius_y_tri[i-1] << " " << std::scientific << Radius_z_tri[i-1] << endl;
    }
    // Now evaluate triangle center of masses and put two points to represent the normals there
    for(int i=0; i<faces; i++) {
        double triangle_com[3] = {0,0,0};
        int dummy_1 = triangle_list[i][0];
        int dummy_2 = triangle_list[i][1];
        int dummy_3 = triangle_list[i][2];
        triangle_com[0] = Radius_x_tri[dummy_1];
        triangle_com[1] = Radius_y_tri[dummy_1];
        triangle_com[2] = (Radius_z_tri[dummy_1]+Radius_z_tri[dummy_2]+Radius_z_tri[dummy_3])/3.0;
        // x
        if((triangle_com[0]-Radius_x_tri[dummy_2]) > 0.5) {
            triangle_com[0] = (triangle_com[0]+(1.0+Radius_x_tri[dummy_2]))/2.0;
        }
        else if((triangle_com[0]-Radius_x_tri[dummy_2]) < -0.5) {
            triangle_com[0] = (triangle_com[0]+(-1.0+Radius_x_tri[dummy_2]))/2.0;
        }
        else {
            triangle_com[0] = (triangle_com[0]+Radius_x_tri[dummy_2])/2.0;
        }
        triangle_com[0] = fmod(fmod(triangle_com[0],1.0)+1.0,1.0);
        if((triangle_com[0]-Radius_x_tri[dummy_3]) > 0.5) {
            triangle_com[0] = (triangle_com[0]*2.0+(1.0+Radius_x_tri[dummy_3]))/3.0;
        }
        else if((triangle_com[0]-Radius_x_tri[dummy_3]) < -0.5) {
            triangle_com[0] = (triangle_com[0]*2.0+(-1.0+Radius_x_tri[dummy_3]))/3.0;
        }
        else {
            triangle_com[0] = (triangle_com[0]*2.0+Radius_x_tri[dummy_3])/3.0;
        }
        triangle_com[0] = fmod(fmod(triangle_com[0],1.0)+1.0,1.0);
        // y
        if((triangle_com[1]-Radius_y_tri[dummy_2]) > 0.5) {
            triangle_com[1] = (triangle_com[1]+(1.0+Radius_y_tri[dummy_2]))/2.0;
        }
        else if((triangle_com[1]-Radius_y_tri[dummy_2]) < -0.5) {
            triangle_com[1] = (triangle_com[1]+(-1.0+Radius_y_tri[dummy_2]))/2.0;
        }
        else {
            triangle_com[1] = (triangle_com[1]+Radius_y_tri[dummy_2])/2.0;
        }
        triangle_com[1] = fmod(fmod(triangle_com[1],1.0)+1.0,1.0);
        if((triangle_com[1]-Radius_y_tri[dummy_3]) > 0.5) {
            triangle_com[1] = (triangle_com[1]*2.0+(1.0+Radius_y_tri[dummy_3]))/3.0;
        }
        else if((triangle_com[1]-Radius_y_tri[dummy_3]) < -0.5) {
            triangle_com[1] = (triangle_com[1]*2.0+(-1.0+Radius_y_tri[dummy_3]))/3.0;
        }
        else {
            triangle_com[1] = (triangle_com[1]*2.0+Radius_y_tri[dummy_3])/3.0;
        }
        triangle_com[1] = fmod(fmod(triangle_com[1],1.0)+1.0,1.0);
        // now get normal
        double normal_com[3] = {0,0,0};
        normalTriangle(i, normal_com);
        // Now get lower and upper points
        double upper_point[3] = {0,0,0};
        double lower_point[3] = {0,0,0};
        upper_point[0] = Length_x*triangle_com[0]+normal_com[0];
        upper_point[1] = Length_y*triangle_com[1]+normal_com[1];
        upper_point[2] = triangle_com[2]+normal_com[2];
        upper_point[0] = fmod(fmod(upper_point[0],Length_x)+Length_x,Length_x);
        upper_point[1] = fmod(fmod(upper_point[1],Length_y)+Length_y,Length_y);
        lower_point[0] = Length_x*triangle_com[0]-normal_com[0];
        lower_point[1] = Length_y*triangle_com[1]-normal_com[1];
        lower_point[2] = triangle_com[2]-normal_com[2];
        lower_point[0] = fmod(fmod(lower_point[0],Length_x)+Length_x,Length_x);
        lower_point[1] = fmod(fmod(lower_point[1],Length_y)+Length_y,Length_y);
        // Now output
        myfile << " " << 3 << " " << std::scientific << upper_point[0] << " " << std::scientific << upper_point[1] << " " << std::scientific << upper_point[2] << "\n";
        myfile << " " << 4 << " " << std::scientific << lower_point[0] << " " << std::scientific << lower_point[1] << " " << std::scientific << lower_point[2] << "\n";
    }
    myfile.close();
}

void OutputSystem::DumpXYZCheckerboard(string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.precision(6);
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << vertices << endl;
    myfile << "# step " << count_step << endl;
    for(int i=1; i <= vertices; i++) {
        myfile << " " << checkerboard_index[i-1] << " " << std::scientific << Length_x*Radius_x_tri[i-1] << " " << std::scientific << Length_y*Radius_y_tri[i-1] << " " << std::scientific << Radius_z_tri[i-1] << "\n";
        // myfile << " " << Ising_Array[i-1] << " " << std::scientific << Radius_x_tri[i-1] << " " << std::scientific << Radius_y_tri[i-1] << " " << std::scientific << Radius_z_tri[i-1] << endl;
    }
    myfile.close();
}

void OutputSystem::DumpPhiNode(string name){
    ofstream myfile;
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << "# Phi at each node at " << steps_tested_displace+steps_tested_tether+steps_tested_mass << endl;
    myfile << "# i Phi" << endl;
    for(int i=1; i <= vertices; i++){
        myfile << i << " " << std::scientific << phi_vertex[i-1] << endl;
    }
    myfile.close();
}

void OutputSystem::DumpAreaNode(string name){
    ofstream myfile;
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << "# Area at each face at " << steps_tested_displace+steps_tested_tether+steps_tested_mass << endl;
    myfile << "# i Area" << endl;
    for(int i=1; i <= faces; i++){
        myfile << i << " " << std::scientific << area_faces[i-1] << endl;
    }
    myfile.close();
}
