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
#include "analyzers.hpp"
#include "init_system.hpp"
#include "output_system.hpp"
#include "simulation.hpp"
#include "saruprng.hpp"
using namespace std;

int main(int argc, char* argv[]) {
    // Initialize MPI
    // Should build in restarting somehow..
    // Start the clock
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    MPI_Init(&argc, &argv);
    // Generate system and read params
    MembraneMC system(omp_get_max_threads());
    system.InputParam(argc, argv);
    // Run initializer
    InitSystem init;
    init.Initialize(system);
    // Analyzers
    Analyzers analysis(system.bins, system.storage_time, system.storage_umb_time, system);
    // Generate neighbor lists
    NeighborList nl;
    nl.GenerateNeighborList(system);
    // Run simulation
    Simulation simulate(system.lambda, system.lambda_scale, system.nl_move_start, omp_get_max_threads());
    // Adjust temperature before equilibration
    if(system.restart == 0) {
        system.temp = system.temp_list[0];
        simulate.Equilibriate(system.cycles_eq, system, nl, begin);
    }
    // Adjust temperature before simulation
    system.temp = system.temp_list[1];
    simulate.Simulate(system.cycles_prod, system, nl, analysis, begin);
    analysis.OutputAnalyzers(system);
    system.OutputTimes(begin);
    // Finalize the MPI environment
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
