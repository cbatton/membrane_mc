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
#include "output_system.hpp"
using namespace std;

OutputSystem::OutputSystem() {
    // Constructor
    // Does nothing
}

OutputSystem::~OutputSystem() {
    // Destructor
    // Does nothing
}

void OutputSystem::OutputTriangulation(MembraneMC& sys, string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.open(sys.output_path+"/"+name);
    myfile.precision(17);
    myfile << "Length_x " << std::scientific << sys.lengths[0] << " Length_y " << std::scientific << sys.lengths[1] << " Length_z " << std::scientific << sys.lengths[2] << " count_step " << sys.count_step << endl;
    myfile << sys.vertices << " " << sys.faces << " " << 0 << endl;
	myfile << endl;
	// Input radius values
	for(int i=0; i<sys.vertices; i++){
		myfile << sys.ising_array[i] << " " << std::scientific << sys.lengths[0]*sys.radii_tri[i][0] << " " << std::scientific << sys.lengths[1]*sys.radii_tri[i][1] << " " << std::scientific << sys.radii_tri[i][2] << "\n";
	}

    // Skip 2 lines
    for(int i=0; i<2; i++){
        myfile << "\n";
    }
	// Input triangle connectivity
	for(int i=0; i<sys.faces; i++){
		myfile << sys.triangle_list[i][0] << " " << sys.triangle_list[i][1] << " " << sys.triangle_list[i][2] << "\n";
	}
    myfile.close();
}

void OutputSystem::OutputTriangulationAppend(MembraneMC& sys, string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.open(sys.output_path+"/"+name, std::ios_base::app);
    myfile << "sys.lengths[0] " << std::scientific << sys.lengths[0] << " sys.lengths[1] " << std::scientific << sys.lengths[1] << " sys.lengths[2] " << std::scientific << sys.lengths[2] << " count_step " << sys.count_step << endl;
    myfile << sys.vertices << " " << sys.faces << " " << 0 << endl;
	myfile << endl;
	for(int i=0; i<sys.vertices; i++){
		myfile << sys.ising_array[i] << " " << std::scientific << sys.lengths[0]*sys.radii_tri[i][0] << " " << std::scientific << sys.lengths[1]*sys.radii_tri[i][1] << " " << std::scientific << sys.radii_tri[i][2] << "\n";
	}

    // Skip 2 lines
    for(int i=0; i<2; i++){
        myfile << "\n";
    }
	// Input triangle connectivity
	for(int i=0; i<sys.faces; i++){
		myfile << sys.triangle_list[i][0] << " " << sys.triangle_list[i][1] << " " << sys.triangle_list[i][2] << "\n";
	}
    myfile.close();
}

void OutputSystem::OutputTriangulationStorage(MembraneMC& sys) {
    ofstream myfile;

    myfile.open (sys.output_path+"/triangle_list.txt", std::ios_base::app);
    for(int i=0; i<sys.faces; i++){
        myfile << "For face " << i << " the points are " << sys.triangle_list[i][0] << " " << sys.triangle_list[i][1] << " " << sys.triangle_list[i][2] << endl;
    }
    myfile.close();

    myfile.open (sys.output_path+"/max_numbers.txt", std::ios_base::app);
    for(int i=0; i<sys.vertices; i++) {
        myfile << "Max neighbors at " << i << " is " << sys.point_neighbor_list[i].size() << endl;
        myfile << "Max triangles at " << i << " is " << sys.point_triangle_list[i].size() << endl;
    }
    myfile.close();

    myfile.open (sys.output_path+"/point_neighbor_list.txt", std::ios_base::app);
    for(int i=0; i<sys.vertices; i++) {
        myfile << "Neighbor list at vertex " << i << " is given by : ";
        for(int j=0; j<sys.point_neighbor_list[i].size(); j++){
            myfile << sys.point_neighbor_list[i][j] << " ";
        }
        myfile << endl;
    }
    myfile.close();

    myfile.open (sys.output_path+"/point_triangle_list.txt", std::ios_base::app);
    for(int i=0; i<sys.vertices; i++) {
        myfile << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<sys.point_triangle_list[i].size(); j++) {
            myfile << sys.point_triangle_list[i][j] << " ";
        }
        myfile << endl;
    }
    myfile.close();

}

void OutputSystem::DumpXYZConfig(MembraneMC& sys, string name) {
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    ofstream myfile;
    myfile.precision(6);
    myfile.open (sys.output_path+"/"+name, std::ios_base::app);
    myfile << sys.vertices << endl;
    myfile << "Lattice=\"" << sys.lengths[0] << " 0.0 0.0 0.0 " << sys.lengths[1] << " 0.0 0.0 0.0 " << sys.lengths[2] << "\" ";
    myfile << "Origin=\"" << -0.5*sys.lengths[0] << " " << -0.5*sys.lengths[1] << " " << -0.5*sys.lengths[2] << "\" ";
    myfile << "Properties=species:S:3:pos:R:3 ";
    myfile << "Time=" << sys.count_step << "\n";
    for(int i=0; i < sys.vertices; i++) {
        myfile << " " << sys.ising_array[i] << " " << std::scientific << sys.lengths[0]*sys.radii_tri[i][0] << " " << std::scientific << sys.lengths[1]*sys.radii_tri[i][1] << " " << std::scientific << sys.radii_tri[i][2] << "\n";
    }
    myfile.close();
}
