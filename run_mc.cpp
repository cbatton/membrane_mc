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
using namespace std;

int main(int argc, char* argv[]) {
    // Initialize MPI
    // Should build in restarting somehow..
    MPI_Init(&argc, &argv);
    MembraneMC system;
    system.InputParams(argc, argv);
    system.Equilibriate(system.cycles_equil);
    system.Simulate(system.cycles);
    system.OutputTimes();
    system.OutputAnalyzers();
    // Finalize the MPI environment
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
