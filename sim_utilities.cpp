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
#include "sim_utilities.hpp"
using namespace std;

double SimUtilities::WrapDistance(double a, double b, double Length){
    //returns the minimum distance between two numbers between 0 and L, sign ignored
    double dx = a-b;
    if(dx > (Length_x*0.5)){
        return dx = dx - Length_x;
    }
    else if(dx <= (-Length_x*0.5)){
        return dx = dx + Length_x;
    }
    return dx;
}

double SimUtilities::LengthLink(int i, int j) {
    // Compute distance between two points
    double distX = wrapDistance_x(Length_x*Radius_x_tri[i], Length_x*Radius_x_tri[j]);
    double distY = wrapDistance_y(Length_y*Radius_y_tri[i], Length_y*Radius_y_tri[j]);
    double distZ = Radius_z_tri[i] - Radius_z_tri[j];
    return pow(pow(distX,2.0) + pow(distY,2.0) + pow(distZ,2.0),0.5);
}

void SimUtilities::AreaNode(int i) {
    // Compute area of a face
    int dummy_1 = triangle_list[i][0];
    int dummy_2 = triangle_list[i][1];
    int dummy_3 = triangle_list[i][2];
    double ac_1 = wrapDistance_x(Length_x*Radius_x_tri[dummy_1], Length_x*Radius_x_tri[dummy_2]);
    double ac_2 = wrapDistance_y(Length_y*Radius_y_tri[dummy_1], Length_y*Radius_y_tri[dummy_2]);
    double ac_3 = Radius_z_tri[dummy_1] - Radius_z_tri[dummy_2];
    double bd_1 = wrapDistance_x(Length_x*Radius_x_tri[dummy_1], Length_x*Radius_x_tri[dummy_3]);
    double bd_2 = wrapDistance_y(Length_y*Radius_y_tri[dummy_1], Length_y*Radius_y_tri[dummy_3]);
    double bd_3 = Radius_z_tri[dummy_1] - Radius_z_tri[dummy_3];

    // Area is equal to 1/2*magnitude(AB cross AC)
    area_faces[i] = 0.5*pow(pow(ac_2*bd_3-ac_3*bd_2,2.0)+pow(-ac_1*bd_3+ac_3*bd_1,2.0)+pow(ac_1*bd_2-ac_2*bd_1,2.0) , 0.5);
}

void SimUtilities::NormalTriangle(int i, double normal[3]) {
    // compute normal of a face
    int dummy_1 = triangle_list[i][0];
    int dummy_2 = triangle_list[i][1];
    int dummy_3 = triangle_list[i][2];
    double ac_1 = wrapDistance_x(Length_x*Radius_x_tri[dummy_1], Length_x*Radius_x_tri[dummy_2]);
    double ac_2 = wrapDistance_y(Length_y*Radius_y_tri[dummy_1], Length_y*Radius_y_tri[dummy_2]);
    double ac_3 = Radius_z_tri[dummy_1] - Radius_z_tri[dummy_2];
    double bd_1 = wrapDistance_x(Length_x*Radius_x_tri[dummy_1], Length_x*Radius_x_tri[dummy_3]);
    double bd_2 = wrapDistance_y(Length_y*Radius_y_tri[dummy_1], Length_y*Radius_y_tri[dummy_3]);
    double bd_3 = Radius_z_tri[dummy_1] - Radius_z_tri[dummy_3];

    // Normal is equal to AC cross BD / magnitude(AC cross BD)
    double magnitude = pow(pow(ac_2*bd_3-ac_3*bd_2,2.0)+pow(-ac_1*bd_3+ac_3*bd_1,2.0)+pow(ac_1*bd_2-ac_2*bd_1,2.0) , 0.5);
    normal[0] = (ac_2*bd_3-ac_3*bd_2)/magnitude;
    normal[1] = (-ac_1*bd_3+ac_3*bd_1)/magnitude;
    normal[2] = (ac_1*bd_2-ac_2*bd_1)/magnitude;
}

void SimUtilities::ShuffleSaru(Saru& saru, vector<int> &vector_int) {
    for(int i=(vector_int.size()-1); i>0; i--) {
        swap(vector_int[i], vector_int[saru.rand_select(i)]);
    }
}

double SimUtilities::CosineAngle(int i, int j, int k) {
    // Compute angle given by ij, ik
    double x1 = wrapDistance_x(Length_x*Radius_x_tri[j], Length_x*Radius_x_tri[i]);
    double y1 = wrapDistance_y(Length_y*Radius_y_tri[j], Length_y*Radius_y_tri[i]);
    double z1 = Radius_z_tri[j] - Radius_z_tri[i];
    double x2 = wrapDistance_x(Length_x*Radius_x_tri[k], Length_x*Radius_x_tri[i]);
    double y2 = wrapDistance_y(Length_y*Radius_y_tri[k], Length_y*Radius_y_tri[i]);
    double z2 = Radius_z_tri[k] - Radius_z_tri[i];
    return acos_fast((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
}

double SimUtilities::CosineAngleNorm(int i, int j, int k) {
    // Compute angle given by ij, ik
    double x1 = wrapDistance_x(Length_x*Radius_x_tri[j], Length_x*Radius_x_tri[i]);
    double y1 = wrapDistance_y(Length_y*Radius_y_tri[j], Length_y*Radius_y_tri[i]);
    double z1 = Radius_z_tri[j] - Radius_z_tri[i];
    double x2 = wrapDistance_x(Length_x*Radius_x_tri[k], Length_x*Radius_x_tri[i]);
    double y2 = wrapDistance_y(Length_y*Radius_y_tri[k], Length_y*Radius_y_tri[i]);
    double z2 = Radius_z_tri[k] - Radius_z_tri[i];
    return acos((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
}

double SimUtilities::Cotangent(double x) {
//    return cos(x)/sin(x);
    return tan(M_PI_2 - x);
}

void SimUtilities::AcosFastInitialize() {    
    for(int i=0; i<look_up_density; i++) {
        lu_points_acos[i] = acos_lu_min+i*lu_interval_acos;
        lu_values_acos[i] = acos(lu_points_acos[i]);
    }
    /*
    ofstream myfile;
    myfile.open ("acos.txt");
    myfile << "This file has the table for acos" << endl;
    for(int i=0; i<look_up_density; i++) {
        myfile << lu_points_acos[i] << " " << std::scientific << lu_values_acos[i] << " " << acos_fast(lu_points_acos[i]) << endl;
    }
    myfile.close();
    */
}

inline double SimUtilities::AcosFast(double x) {
    double look_up = (x-acos_lu_min)*in_lu_interval_acos;
    int look_up_value = floor(look_up);
    return lu_values_acos[look_up_value]+(lu_values_acos[look_up_value+1]-lu_values_acos[look_up_value])*(look_up-look_up_value);
}

void SimUtilities::CotangentFastInitialize() {
    for(int i=0; i<look_up_density; i++) {
        lu_points_cot[i] = cot_lu_min+i*lu_interval_cot;
        lu_values_cot[i] = cotangent(lu_points_cot[i]);
    }
    /*
    ofstream myfile;
    myfile.open ("cot.txt");
    myfile << "This file has the table for cot" << endl;
    for(int i=0; i<look_up_density; i++) {
        myfile << lu_points_cot[i] << " " << std::scientific << lu_values_cot[i] << " " << cotangent_fast(lu_points_cot[i]) << endl;
    }
    myfile.close();
    */
}

inline double SimUtilities::CotangentFast(double x) {
    double look_up = (x-cot_lu_min)*in_lu_interval_cot;
    int look_up_value = floor(look_up);
    return lu_values_cot[look_up_value]+(lu_values_cot[look_up_value+1]-lu_values_cot[look_up_value])*(look_up-look_up_value);
}
