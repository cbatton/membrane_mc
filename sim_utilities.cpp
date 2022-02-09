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

SimUtilities::SimUtilities(MembraneMC* sys_) {
    // Constructor
    // Assign system to current system
    sys = sys_;
}

SimUtilities::~SimUtilities() {
    // Destructor
    // Does nothing
}

double SimUtilities::WrapDistance(double a, double b){
    // Performs PBC
    // As working on frame of (-0.5,0,5], just need to round
    double dx = a-b;
    return dx-round(dx);
}

double SimUtilities::LengthLink(int i, int j) {
    // Compute distance between two points
    double distX = sys->lengths[0]*WrapDistance(sys->radii_tri[i][0], sys->radii_tri[j][0]);
    double distY = sys->lengths[1]*WrapDistance(sys->radii_tri[i][1], sys->radii_tri[j][1]);
    double distZ = sys->radii_tri[i][2] - sys->radii_tri[j][2];
    return pow(pow(distX,2.0) + pow(distY,2.0) + pow(distZ,2.0),0.5);
}

void SimUtilities::AreaNode(int i) {
    // Compute area of a face
    int dummy_1 = sys->triangle_list[i][0];
    int dummy_2 = sys->triangle_list[i][1];
    int dummy_3 = sys->triangle_list[i][2];
    double ac_1 = sys->lengths[0]*WrapDistance(sys->radii_tri[dummy_1][0], sys->radii_tri[dummy_2][0]);
    double ac_2 = sys->lengths[1]*WrapDistance(sys->radii_tri[dummy_1][1], sys->radii_tri[dummy_2][1]);
    double ac_3 = sys->radii_tri[dummy_1][2] - sys->radii_tri[dummy_2][2];
    double bd_1 = sys->lengths[0]*WrapDistance(sys->radii_tri[dummy_1][0], sys->radii_tri[dummy_3][0]);
    double bd_2 = sys->lengths[1]*WrapDistance(sys->radii_tri[dummy_1][1], sys->radii_tri[dummy_3][1]);
    double bd_3 = sys->radii_tri[dummy_1][2] - sys->radii_tri[dummy_3][2];

    // Area is equal to 1/2*magnitude(AB cross AC)
    area_faces[i] = 0.5*pow(pow(ac_2*bd_3-ac_3*bd_2,2.0)+pow(-ac_1*bd_3+ac_3*bd_1,2.0)+pow(ac_1*bd_2-ac_2*bd_1,2.0) , 0.5);
}

void SimUtilities::NormalTriangle(int i, double normal[3]) {
    // compute normal of a face
    int dummy_1 = sys->triangle_list[i][0];
    int dummy_2 = sys->triangle_list[i][1];
    int dummy_3 = sys->triangle_list[i][2];
    double ac_1 = sys->lengths[0]*WrapDistance(sys->radii_tri[dummy_1][0], sys->radii_tri[dummy_2][0]);
    double ac_2 = sys->lengths[1]*WrapDistance(sys->radii_tri[dummy_1][1], sys->radii_tri[dummy_2][1]);
    double ac_3 = sys->radii_tri[dummy_1][2] - sys->radii_tri[dummy_2][2];
    double bd_1 = sys->lengths[0]*WrapDistance(sys->radii_tri[dummy_1][0], sys->radii_tri[dummy_3][0]);
    double bd_2 = sys->lengths[1]*WrapDistance(sys->radii_tri[dummy_1][1], sys->radii_tri[dummy_3][1]);
    double bd_3 = sys->radii_tri[dummy_1][2] - sys->radii_tri[dummy_3][2];

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
    double ac_1 = sys->lengths[0]*WrapDistance(sys->radii_tri[j][0], sys->radii_tri[i][0]);
    double ac_2 = sys->lengths[1]*WrapDistance(sys->radii_tri[j][1], sys->radii_tri[i][1]);
    double ac_3 = sys->radii_tri[j][2] - sys->radii_tri[i][2];
    double bd_1 = sys->lengths[0]*WrapDistance(sys->radii_tri[k][0], sys->radii_tri[i][0]);
    double bd_2 = sys->lengths[1]*WrapDistance(sys->radii_tri[k][1], sys->radii_tri[i][1]);
    double bd_3 = sys->radii_tri[k][2] - sys->radii_tri[i][2];
    double dot = ac_1*bd_1+ac_2*bd_2+ac_3*bd_3;
    double cross = sqrt(pow(ac_2*bd_3-ac_3*bd_2,2)+pow(-ac_1*bd_3+ac_3*bd_1,2)+pow(ac_1*bd_2-ac_2*bd_1,2));
    return dot/cross;
}
