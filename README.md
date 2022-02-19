# membrane_mc
A C++ code implementing a parallelized Monte Carlo simulation of a triangulated surface model. 
The triangulated surface model captures the large-scale energetics of a two-phased lipid bilayer that has protein domains that favor one phase.
It does so by simulating the bending energy per the Helfrich model, and the composition energy by an Ising model where the lipid species are treated as non-mass conserving and the protein is mass conserving.
Currently a planar membrane with periodic boundary conditions is simulated, though it is relatively simple to implement any domain possible as long as a triangulation is available.
Monte Carlo moves are implemented that displace vertices (DisplaceStep), change the triangulation by flipping a bond (TetherCut), identity changes of the lipid species (ChangeMassNonCon), mass conserving non-local swap of identities of protein and lipid species (MoveProteinNL), and changing the area of the XY-plane.
All moves are accepted using the Metropolis criterion.

The code uses OpenMP to parallelize the Monte Carlo calculations using the checkerboard scheme of Glotzer (https://arxiv.org/abs/1211.1646).
MPI is used in a primitive form to run multiple copies in parallel that read different inputs.
The code allows for automatic restarting.
The simulation output can be visualized using any program that reads xyz files.

Listed files:
- `run_mc.cpp`, uses all the pieces to drive the simulation
- `membrane_mc.cpp`, implements the main data structures
- `init_system.cpp`, contains functions that initialize the system
- `output_system.cpp`, contains functions that output the system in readable formats
- `neighborlist.cpp`, implements neighbor list and checkerboard procedure
- `mc_moves.cpp`, implements Monte Carlo moves
- `utilities.cpp`, implements utility functions used throughout the over files
- `simulation.cpp`, implements main equilibration and simulation procedures

# Output examples

Example of a system where the proteins have no spontaneous curvature, leading to macrophase separation.
![macro](figures/multiple_proteins_C_1.png)
Example of a system where the proteins have spontaneous curvature, leading to microphase separation.
![micro](figures/multiple_proteins_C_5.png)
