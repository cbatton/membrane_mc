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

// Initial mesh is points distributed in rectangular grid
const int dim_x = 200; // Nodes in x direction
const int dim_y = 200; // Nodes in y direction
// Triangle properties
const int vertices = dim_x*dim_y;
const int faces = 2*dim_x*dim_y;
int active_vertices = vertices;
int active_faces = faces;

// Note total number of nodes is dim_x*dim_y
double Radius_x[dim_x][dim_y]; // Array to store x coordinate of node
double Radius_y[dim_x][dim_y]; // Array to store y coordinate of node
double Radius_z[dim_x][dim_y]; // Array to store z coordinate of node
double Radius_x_original[dim_x][dim_y];
double Radius_y_original[dim_x][dim_y];
double Radius_z_original[dim_x][dim_y];
// Triangulation radius values
double Radius_x_tri[vertices];
double Radius_y_tri[vertices];
double Radius_z_tri[vertices];
double Radius_x_tri_original[vertices];
double Radius_y_tri_original[vertices];
double Radius_z_tri_original[vertices];
int Ising_Array[vertices];
// Triangles
const int neighbor_min = 2;
const int neighbor_max = 10;
int triangle_list[faces][3];
int point_neighbor_max[vertices];
int point_neighbor_list[vertices][neighbor_max];
int point_triangle_list[vertices][neighbor_max];
int point_triangle_max[vertices];
int point_neighbor_triangle[vertices][neighbor_max][2];

int triangle_list_original[faces][3];
int point_neighbor_list_original[vertices][neighbor_max];
int point_neighbor_max_original[vertices];
int point_triangle_list_original[vertices][neighbor_max];
int point_triangle_max_original[vertices];
int point_neighbor_triangle_original[vertices][neighbor_max][2];

// Neighborlist
vector<vector<int>> neighbor_list; // Neighbor list
vector<int> neighbor_list_index; // Map from particle to index
vector<vector<int>> neighbors; // Neighboring bins
vector<int> index_particles_max_nl;
// Indexing for neighbor list
int nl_x = int(9000*2.0)/1.00-1;
int nl_y = int(9000*2.0)/1.00-1;
int nl_z = int(60*2.0)/1.00-1;

// Checkerboard set
vector<vector<int>> checkerboard_list; // Neighbor list
vector<int> checkerboard_index; // Map from particle to index
vector<vector<int>> checkerboard_neighbors; // Neighboring bins
// Indexing for neighbor list
int checkerboard_x = 1;
int checkerboard_y = 1;
double checkerboard_set_size = 3.5;

/*
Rambling ideas included in comments
Essentially, because I'm not that good at CGAL yet,
CGAL will be used to generate the needed triangulation and then outputted
into a file. From the file, the needed connectivity info will be read. Each point
will then be read into a neighbor list in order to determine what links are changed by displacement
Links will be common to two triangles, so a list of triangles that each link is common to allow
for efficent computation of that quantity. Now, for how the link list will be handled.......
1. Search triangle list for pairs, store in format with lowered number point, higher numbered point
by dim_x value


Switched triangle radius values to use the vertices index instead of dim_x*dim_y, as that
is more natural with respect to the problem
*/
double phi_vertex[vertices];
double phi_vertex_original[vertices];
double mean_curvature_vertex[vertices];
double mean_curvature_vertex_original[vertices];
double sigma_vertex[vertices];
double sigma_vertex_original[vertices];
double Phi = 0.0; // Energy at current step
double Phi_phi = 0.0; // Composition energy at current step
double Phi_bending = 0.0; // Bending energy at current step
int Mass = 0;
double Magnet = 0;
double area_faces[faces];
double area_faces_original[faces];
double Area_total;
double sigma_i_total = 0.0;
double Area_proj_average;

double k_b[3] = {20.0, 20.0, 20.0}; // k units
double k_g[3] = {0.0, 0.0, 0.0}; // k units
double k_a[3] = {0.0, 0.0, 0.0}; // k units
double gamma_surf[3] = {0.0, 0.0, 0.0}; // k units
double tau_frame = 0;
double ising_values[3] = {-1.0, 1.0, 1.0};
double spon_curv[3] = {0.0,0.0,0.0}; // Spontaneous curvature at protein nodes
double spon_curv_end = 0.0;
double J_coupling[3][3] = {{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}};
double h_external = 0.0;
// double area_original_total = 0;
// double area_current_total = 0;
double Length_x = 96; // Length of a direction
double Length_y = 96;
double Length_z = 60;
double Length_x_old = Length_x;
double Length_y_old = Length_y;
double Length_z_old = Length_z;
double scale_xy = 1.0;
double Length_x_base = Length_x;
double Length_y_base = Length_y;
double scale_xy_old = scale_xy;
double box_x = Length_x/double(nl_x); 
double box_y = Length_y/double(nl_y); 
double box_z = 2*Length_z/double(nl_z);
double box_x_checkerboard = Length_x/checkerboard_x;
double box_y_checkerboard = Length_y/checkerboard_y;
double cell_center_x = 0.0;
double cell_center_y = 0.0;
double lambda = 0.0075; // Maximum percent change in displacement
double lambda_scale = 0.01; // Maximum percent change in scale
double num_frac = 0.5; // Number fraction of types
double scale = 1.0;
const int Cycles_eq = 1000001;
const int Cycles_prod = 1000001;
double T = 2.0; // Temperature
double T_list[2] = {2.0, 2.0};
long long int steps_tested_displace = 0;
long long int steps_rejected_displace = 0;
long long int steps_tested_tether = 0;
long long int steps_rejected_tether = 0;
long long int steps_tested_mass = 0;
long long int steps_rejected_mass = 0;
long long int steps_tested_protein = 0;
long long int steps_rejected_protein = 0;
long long int steps_tested_area = 0;
long long int steps_rejected_area = 0;
long long int steps_tested_eq = 0;
long long int steps_rejected_eq = 0;
long long int steps_tested_prod = 0;
long long int steps_rejected_prod = 0;
// nl move parameter
int nl_move_start = 0;

// Protein variables
int protein_node[vertices];
int num_proteins = 2;

// Storage variables
const int storage_time = 10;
const int storage = Cycles_prod/storage_time;
int storage_counts = 0;
double energy_storage[storage+1];
double area_storage[storage+1];
double area_proj_storage[storage+1];
double mass_storage[storage+1];
// Analyzer
const int storage_neighbor =  10;
const int storage_umb_time = 100;
const int storage_umb = Cycles_prod/storage_umb_time;
int umb_counts = 0;
int neighbor_counts = 0;
long long int numbers_neighbor[neighbor_max][Cycles_prod/storage_neighbor];
double energy_storage_umb[storage_umb];
double energy_harmonic_umb[storage_umb];

// Pseudo-random number generators
vector<Saru> generators;
Saru generator;
unsigned int seed_base = 0;
unsigned int count_step = 0;

// Time
chrono::steady_clock::time_point t1;
chrono::steady_clock::time_point t2;
double final_time = 120.0;
double final_warning = 60.0;
double time_storage_cycle[2] = {0,0};
double time_storage_area[4] = {0,0,0,0};
double time_storage_displace[3] = {0,0,0};
double time_storage_other[8] = {0,0,0,0,0,0,0,0};
double time_storage_overall;

// Look up table variables
const int look_up_density = 10001;
double acos_lu_min = -1.0;
double acos_lu_max = 1.0;
double cot_lu_min = 0.15;
double cot_lu_max = M_PI-0.15;
double lu_interval_acos = (acos_lu_max-acos_lu_min)/(look_up_density-1);
double lu_interval_cot = (cot_lu_max-cot_lu_min)/(look_up_density-1);
double in_lu_interval_acos = 1.0/lu_interval_acos;
double in_lu_interval_cot = 1.0/lu_interval_cot;

// Arrays to store look up tables
double lu_points_acos[look_up_density];
double lu_values_acos[look_up_density];
double lu_points_cot[look_up_density];
double lu_values_cot[look_up_density];

// MPI variables
int world_size=1;
int world_rank=0;
string local_path;
vector<string> rank_paths;
string output;
string output_path;
streambuf *myfilebuf;

// Testing
double max_diff = -1;
double relative_diff = 0;

void initializeState();
void initializeEquilState();
void inputParam();
void inputState();
void SaruSeed(unsigned int);
void generateTriangulation();
void generateTriangulationEquil();
inline int link_triangle_test(int, int);
inline void link_triangle_face(int, int, int *);
void useTriangulation(string);
void useTriangulationRestart(string);
void outputTriangulation(string);
void outputTriangulationAppend(string);
void outputTriangulationStorage();
double wrapDistance_x(double, double);
double wrapDistance_y(double, double);
double lengthLink(int, int);
void generateNeighborList();
void areaNode(int);
void normalTriangle(int i, double normal[3]);
void shuffle_saru(Saru&, vector<int>&);
void generateCheckerboard();
double cosineAngle(int, int, int);
double cosineAngle_norm(int, int, int);
double cotangent(double);
void acos_fast_initialize();
inline double acos_fast(double);
void cotangent_fast_initialize();
inline double cotangent_fast(double);
void linkMaxMin();
void energyNode(int);
void initializeEnergy();
void initializeEnergy_scale();
void energyNode_i(int);
void initializeEnergy_i();
void DisplaceStep(int = -1, int = 0);
void TetherCut(int = -1, int = 0);
void ChangeMassNonCon(int = -1, int = 0);
void ChangeMassNonCon_gl(int = -1);
void ChangeMassCon(int = -1, int = 0);
void ChangeMassCon_nl();
void moveProtein_gen(int, int);
void moveProtein_nl(int, int, int);
void ChangeArea();
void CheckerboardMCSweep(bool);
void nextStepSerial();
void nextStepParallel(bool);
void dumpXYZConfig(string);
void dumpXYZConfigNormal(string);
void dumpXYZCheckerboard(string);
void dumpPhiNode(string);
void dumpAreaNode(string);
void sampleNumberNeighbors(int);
void dumpNumberNeighbors(string, int);
void equilibriate(int, chrono::steady_clock::time_point&);
void simulate(int, chrono::steady_clock::time_point&);
void energyAnalyzer();
void areaAnalyzer();
void areaProjAnalyzer();
void massAnalyzer();
void numberNeighborsAnalyzer();
void umbAnalyzer();
void umbOutput(int, ofstream&);

// openmp stuff
const int max_threads = 272;
int active_threads = 0;
double Phi_diff_thread[max_threads][8];
double Phi_phi_diff_thread[max_threads][8];
double Phi_bending_diff_thread[max_threads][8];
double Area_diff_thread[max_threads][8];
int Mass_diff_thread[max_threads][8];
double Magnet_diff_thread[max_threads][8];
int steps_tested_displace_thread[max_threads][8];
int steps_rejected_displace_thread[max_threads][8];
int steps_tested_tether_thread[max_threads][8];
int steps_rejected_tether_thread[max_threads][8];
int steps_tested_mass_thread[max_threads][8];
int steps_rejected_mass_thread[max_threads][8];
int steps_tested_protein_thread[max_threads][8];
int steps_rejected_protein_thread[max_threads][8];
/*
int omp_get_max_threads();
int omp_get_thread_num();
int omp_get_num_threads();
*/

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // Get local path
    string path_file(argv[1]);
    ifstream input_0;
    input_0.open(path_file);
    input_0 >> local_path;
    input_0.close();
    // Get path for each processor
    string path_files(argv[2]);
    rank_paths.resize(world_size);
    ifstream input_1;
    string line;
    input_1.open(path_files);
    for(int i=0; i<world_size; i++) {
        input_1 >> rank_paths[i];
        getline(input_1,line);
    }
    input_1.close();
    // Print off a hello world message from all processors
    string output = local_path+rank_paths[world_rank]+"/out";
    output_path = local_path+rank_paths[world_rank];
    ofstream myfile;
    myfile.open(output);
    // Redirect cout
    streambuf *coutbuf = cout.rdbuf();
    cout.rdbuf(myfile.rdbuf());
    myfilebuf = myfile.rdbuf();
    #pragma omp parallel
    {
    int thread_max = omp_get_num_threads();
    int thread = omp_get_thread_num();
    #pragma omp critical
    {
    cout << "Hello world from thread " << thread << " out of " << thread_max << " from rank " << world_rank << " out of " << world_size << " processes, output in " << output << "\n";
    }
    }

    // Initialize OpenMP
    active_threads = omp_get_max_threads();
    // Parallel hello world
    /*
    #pragma omp parallel for
    for(int i=0; i<omp_get_num_threads(); i++) {
        #pragma omp critical
        {
        cout << "Hello from thread " << omp_get_thread_num() << endl;
        }
    }
    */
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    // Initialize random number generators
    for(int i=0; i<omp_get_max_threads(); i++) {
        generators.push_back(Saru());
    }

    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    t1_other = chrono::steady_clock::now();
    // Initialize look up tables
    acos_fast_initialize();
    cotangent_fast_initialize();
    inputParam(); // Input param if file present
	// Insert first actin particle in middle of simulation
    cout << "Generate initial state" << endl;
    initializeEquilState();
    
    cout << "Generate triangulation" << endl;
    generateTriangulationEquil();
    // inputState(); // Input configuration if file present
    cout << "Using triangulation" << endl;
    useTriangulation("out.off");
	// Generate neighbor list
    cout << "Neighbor list" << endl;
    generateNeighborList();
    cout << "Checkerboard" << endl;
    generateCheckerboard();
    // Initialize proteins
    // Initialize protein_node
    #pragma omp parallel for
    for(int i=0; i<vertices; i++) {
        protein_node[i] = -1;
    }
    // Place proteins at random
    for(int i=0; i<num_proteins; i++) {
        bool check_nodes = true;
        while(check_nodes) {
            int j = generator.rand_select(vertices-1);
            if(Ising_Array[j] != 2) {
                check_nodes = false;
                Ising_Array[j] = 2;
                protein_node[j] = 0;
            }
        }
    }
    cout << "Total protein size: " << num_proteins << endl;
    cout << "Using triangulation" << endl;
    // useTriangulationRestart("int.off");
	// Generate neighbor list
    cout << "Neighbor list" << endl;
    generateNeighborList();
    cout << "Checkerboard" << endl;
    generateCheckerboard();
  
    dumpXYZConfig("config_initial.xyz");
    t2_other = chrono::steady_clock::now();
    chrono::duration<double> time_span_other = t2_other-t1_other;
    time_storage_other[4] += time_span_other.count();

    t1_other = chrono::steady_clock::now();
	linkMaxMin();
    initializeEnergy();
    cout.rdbuf(myfilebuf);
    cout << "Initial values" << endl;
    cout << "Energy is " << Phi << endl;
    cout << "Mass is " << Mass << endl;
    cout << "Area is " << Area_total << " and " << Length_x*Length_y << endl;
    // cout << "Initial sigma_i_total is: " << sigma_i_total << endl;
    // cout << "Initial phi is: " << Phi << endl;
	T = T_list[0];
	cout << "T is " << T << endl;
    // CheckerboardMCSweep();
    /*
    for(int i=0; i<10000; i++) {
        ChangeArea();
        linkMaxMin();
        if(i%100==0) {
            dumpXYZConfig("config_area.xyz");
        }
    }
    */
	// Loop through extend and displace step and check to make sure it is working
    /*
	for(int i=0; i<1000000; i++) {
		DisplaceStep();
		extend();
		if(i%10000==0) {
			dumpXYZConfig("config_check.xyz");
		}
	}
	*/
	double Phi_ = Phi;
    int Mass_ = Mass;
    sigma_i_total = 0.0;    
    initializeEnergy_i();
    cout.rdbuf(myfilebuf);
	cout << "Phi " << Phi << endl;
	cout << "Difference " << std::scientific << Phi - Phi_ << endl;
	cout << "Sigma_i_total " << sigma_i_total << endl;
	cout << "Area total " << Area_total << endl;
	cout << "Difference " << std::scientific << sigma_i_total - Area_total << endl;
    cout << "Mass difference " << Mass-Mass_ << endl;
    t2_other = chrono::steady_clock::now();
    time_span_other = t2_other-t1_other;
    time_storage_other[5] += time_span_other.count();

    equilibriate(Cycles_eq, begin); // Equilibration run
    cout.rdbuf(myfilebuf);
    cout << "Percentage of rejected steps during equilibration: " << steps_rejected_eq << "/" << steps_tested_eq << endl; 
	//T = T_list[1];
	//cout << "T is " << T << endl;
    //simulate(Cycles_prod, begin); // Production run
    //cout.rdbuf(myfilebuf);
    cout << "Percentage of rejected steps during production: " << steps_rejected_prod << "/" << steps_tested_prod << endl;
    // Dump analyzers and final configuration
    // dumpConfig("Oconfig");
    // dumpXYZConfig("config.xyz");
    t1_other = chrono::steady_clock::now();
    //energyAnalyzer();
    //areaAnalyzer();
    //areaProjAnalyzer();
    //massAnalyzer();
    // numberNeighborsAnalyzer();
	outputTriangulation("final.off");
    // dumpXYZConfig("config_final.xyz");
    t2_other = chrono::steady_clock::now();
    time_span_other = t2_other-t1_other;
    time_storage_other[6] += time_span_other.count();
	// Verification
	// Run steps, and use initializeEnergy to test the results
	// double max_diff = -1;
	// for(int i=0; i<1000000; i++) {
    	// ChangeDisplaceStep();
    	// ChangeMassNonCon();
    	// ChangeMassCon();
        // ChangeMassCon_nl();
		// double Phi_ = Phi;
		// initializeEnergy();
		// cout.precision(17);
		// cout << std::scientific << (Phi - Phi_) << endl;
		// if(abs(Phi - Phi_) > max_diff) {
		// 	max_diff = abs(Phi - Phi_);
		// }
	// }
	// cout << "Max difference " << max_diff << endl;
    t1_other = chrono::steady_clock::now();
	linkMaxMin();
    cout.precision(18);
    cout.rdbuf(myfilebuf);
	Phi_ = Phi;
    Mass_ = Mass;
    sigma_i_total = 0.0;    
    initializeEnergy_i();
    cout.rdbuf(myfilebuf);
	cout << "Phi " << Phi << endl;
	cout << "Difference " << std::scientific << Phi - Phi_ << endl;
	cout << "Sigma_i_total " << sigma_i_total << endl;
	cout << "Area total " << Area_total << endl;
	cout << "Difference " << std::scientific << sigma_i_total - Area_total << endl;
    cout << "Mass difference " << Mass-Mass_ << endl;
    initializeEnergy_scale();
    cout.rdbuf(myfilebuf);
	cout << "Phi " << Phi << endl;
	cout << "Difference " << std::scientific << Phi - Phi_ << endl;
	cout << "Sigma_i_total " << sigma_i_total << endl;
	cout << "Area total " << Area_total << endl;
	cout << "Difference " << std::scientific << sigma_i_total - Area_total << endl;
    cout << "Mass difference " << Mass-Mass_ << endl;
    t2_other = chrono::steady_clock::now();
    time_span_other = t2_other-t1_other;
    time_storage_other[7] += time_span_other.count();
    // Time analysis
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> time_span = end-begin;
    cout << "Total time: " << time_span.count() << " s" << endl;
    cout << "Checkerboard MC time: " << time_storage_cycle[0] << " s" << endl;
    cout << "Area MC time: " << time_storage_cycle[1] << " s" << endl;
    cout << "Other time: " << time_span.count()-time_storage_cycle[0]-time_storage_cycle[1] << " s" << endl;

    cout << "\nDisplace MC breakdown" << endl;
    double displace_storage_total = 0;
    for(int i=0; i<3; i++) {
        displace_storage_total += time_storage_displace[i];
    }
    cout << "Checkerboard: " << time_storage_displace[0] << " s, " << time_storage_displace[0]/time_storage_cycle[1] << " per" << endl;
    cout << "Shuffling: " << time_storage_displace[1] << " s, " << time_storage_displace[1]/time_storage_cycle[1] << " per" << endl;
    cout << "Steps: " << time_storage_displace[2] << " s, " << time_storage_displace[2]/time_storage_cycle[1] << " per" << endl;
    cout << "Misc: " << time_storage_cycle[0]-displace_storage_total << " s, " << (time_storage_cycle[0]-displace_storage_total)/time_storage_cycle[0] << " per" << endl;

    cout << "\nArea MC breakdown" << endl;
    double area_storage_total = 0;
    for(int i=0; i<4; i++) {
        area_storage_total += time_storage_area[i];
    }
    cout << "Initial: " << time_storage_area[0] << " s, " << time_storage_area[0]/time_storage_cycle[1] << " per" << endl;
    cout << "Neighbor list: " << time_storage_area[1] << " s, " << time_storage_area[1]/time_storage_cycle[1] << " per" << endl;
    cout << "Energy calculation: " << time_storage_area[2] << " s, " << time_storage_area[2]/time_storage_cycle[1] << " per" << endl;
    cout << "A/R: " << time_storage_area[3] << " s, " << time_storage_area[3]/time_storage_cycle[1] << " per" << endl;
    cout << "Misc: " << time_storage_cycle[1]-area_storage_total << " s, " << (time_storage_cycle[1]-area_storage_total)/time_storage_cycle[1] << " per" << endl;

    double other_storage_total = 0;
    for(int i=0; i<8; i++) {
        other_storage_total += time_storage_other[i];
    }
    cout << "\nOther breakdown" << endl;
    cout << "Generating system: " << time_storage_other[4] <<" s, " << time_storage_other[4]/other_storage_total << " per" << endl;
    cout << "Initial energy and verification: " << time_storage_other[5] <<" s, " << time_storage_other[5]/other_storage_total << " per" << endl;
    cout << "Initial cycle output: " << time_storage_other[0] <<" s, " << time_storage_other[0]/other_storage_total << " per" << endl;
    cout << "Distance check: " << time_storage_other[1] <<" s, " << time_storage_other[1]/other_storage_total << " per" << endl;
    cout << "Output reject per: " << time_storage_other[2] <<" s, " << time_storage_other[2]/other_storage_total << " per" << endl;
    cout << "Output configurations: " << time_storage_other[3] <<" s, " << time_storage_other[3]/other_storage_total << " per" << endl;
    cout << "Analyzers: " << time_storage_other[6] <<" s, " << time_storage_other[4]/other_storage_total << " per" << endl;
    cout << "Verification: " << time_storage_other[7] <<" s, " << time_storage_other[7]/other_storage_total << " per" << endl;
    cout << "Misc: " << time_span.count()-time_storage_cycle[0]-time_storage_cycle[1]-other_storage_total <<" s";
    // Finalize the MPI environment
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

void initializeState() {
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

void initializeEquilState() {
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

void inputParam() { // Takes parameters from a file named param
    ifstream input;
    input.open(output_path+"/param");
    cout.rdbuf(myfilebuf);
    // Check to see if param present. If not, do nothing
    if (input.fail()) {
        cout << "No input file." << endl;
    }

    else{
        char buffer;
        string line;
        cout << "Param file detected. Changing values." << endl;
        input >> line >> T_list[0] >> T_list[1];
        cout << "T is now " << T_list[0] << " " << T_list[1] << endl;
        getline(input, line);
        input >> line >> k_b[0] >> k_b[1] >> k_b[2];
        cout << "k_b is now " << k_b[0] << " " << k_b[1] << " " << k_b[2] << endl;
        getline(input, line);
        input >> line >> k_g[0] >> k_g[1] >> k_g[2];
        cout << "k_g is now " << k_g[0] << " " << k_g[1] << " " << k_g[2] << endl;
        getline(input, line);
        input >> line >> k_a[0] >> k_a[1] >> k_a[2];
        cout << "k_a is now " << k_a[0] << " " << k_a[1] << " " << k_a[2] << endl;
        getline(input, line);
        input >> line >> gamma_surf[0] >> gamma_surf[1] >> gamma_surf[2];
        cout << "gamma_surf is now " << gamma_surf[0] << " " << gamma_surf[1] << " " << gamma_surf[2] << endl;
        getline(input, line);
        input >> line >> tau_frame;
        cout << "tau_frame is now " << tau_frame << endl;
        getline(input, line);
        input >> line >> spon_curv[0] >> spon_curv[1] >> spon_curv[2]; 
        cout << "Spontaneous curvature is now " << spon_curv[0] << " " << spon_curv[1] << " " << spon_curv[2] << endl;
        getline(input, line);
        input >> line >> Length_x;
        cout << "Length_x is now " << Length_x << endl;
        getline(input, line);
        input >> line >> Length_y;
        cout << "Length_y is now " << Length_y << endl;
        getline(input, line);
        input >> line >> Length_z;
        cout << "Length_z is now " << Length_z << endl;
        getline(input, line);
        input >> line >> lambda;
        cout << "lambda is now " << lambda << endl;
        getline(input, line);
        input >> line >> lambda_scale;
        cout << "lambda_scale is now " << lambda_scale << endl;
        getline(input, line);
        input >> line >> scale;
        cout << "scale is now " << scale << endl;
        getline(input, line);
        input >> line >> ising_values[0] >> ising_values[1] >> ising_values[2];
        cout << "ising_values is now " << ising_values[0] << " " << ising_values[1] << " " << ising_values[2] << endl;
        getline(input, line);
        input >> line >> J_coupling[0][0] >> J_coupling[0][1] >> J_coupling[0][2]; 
        getline(input, line);
        input >> J_coupling[1][0] >> J_coupling[1][1] >> J_coupling[1][2];
        getline(input, line);
        input >> J_coupling[2][0] >> J_coupling[2][1] >> J_coupling[2][2];
        cout << "J is now " << J_coupling[0][0] << " " << J_coupling[0][1] << " " << J_coupling[0][2] << endl;
        cout << "\t" << J_coupling[1][0] << " " << J_coupling[1][1] << " " << J_coupling[1][2] << endl;
        cout << "\t" << J_coupling[2][0] << " " << J_coupling[2][1] << " " << J_coupling[2][2] << endl;
        getline(input, line);
        input >> line >> h_external;
        cout << "h is now " << h_external << endl;
        getline(input, line);
        input >> line >> num_frac;
        cout << "num_frac is now " << num_frac << endl;
        getline(input, line);
        input >> line >> num_proteins;
        cout << "num_proteins is now " << num_proteins << endl;
        getline(input, line);
        input >> line >> seed_base;
        cout << "seed_base is now " << seed_base << endl;
        getline(input, line);
        input >> line >> count_step;
        cout << "count_step is now " << count_step << endl;
        getline(input, line);
        input >> line >> final_time;
        cout << "final_time is now " << final_time << endl;
        getline(input, line);
        input >> line >> nl_move_start;
        cout << "nl_move_start is now " << nl_move_start << endl;
        input.close();
        Length_x_old = Length_x;
        Length_y_old = Length_y;
        Length_z_old = Length_z;
        Length_x_base = Length_x;
        Length_y_base = Length_y;
        // Hash seed_base
        seed_base = seed_base*0x12345677 + 0x12345;
        seed_base = seed_base^(seed_base>>16);
        seed_base = seed_base*0x45679;
        cout << "seed_base is now " << seed_base << endl;
        SaruSeed(count_step);
        final_warning = final_time-60.0;
        spon_curv_end = spon_curv[2];
        spon_curv[2] = 0.0;
    }
}

void inputState() {
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

void SaruSeed(unsigned int value) {
// Prime Saru with input seeds of seed_base, value, and OpenMP threads
    generator = Saru(seed_base, value);
    #pragma omp parallel for
    for(int i=0; i<omp_get_max_threads(); i++) {
        generators[i] = Saru(seed_base, value, i);
    }
}

void generateTriangulation() {
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

void generateTriangulationEquil() {
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

inline int link_triangle_test(int vertex_1, int vertex_2) {
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

inline void link_triangle_face(int vertex_1, int vertex_2, int face[2]) {
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

void useTriangulation(string name) {
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

void useTriangulationRestart(string name) {
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
        Length_x_base = Length_x;
        Length_y_base = Length_y;
        // Skip 2 lines
        for(int i=0; i<2; i++){
            getline(input,line);
        } 
        // Input radius values
		double scale_factor = pow(scale,0.5);
        for(int i=0; i<active_vertices; i++){
            input >> Ising_Array[i] >> Radius_x_tri[i] >> Radius_y_tri[i] >> Radius_z_tri[i];
            if(Ising_Array[i] != 2) {
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

void outputTriangulation(string name) {
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

void outputTriangulationAppend(string name) {
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

void outputTriangulationStorage() {
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

double wrapDistance_x(double a, double b){
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


double wrapDistance_y(double a, double b){
    //returns the minimum distance between two numbers between 0 and L, sign ignored
    double dx = a-b;
    if(dx > (Length_y*0.5)){
        return dx = dx - Length_y;
    }
    else if(dx <= (-Length_y*0.5)){
        return dx = dx + Length_y;
    }
    return dx;
}

double lengthLink(int i, int j) {
    // Compute distance between two points
    double distX = wrapDistance_x(Length_x*Radius_x_tri[i], Length_x*Radius_x_tri[j]);
    double distY = wrapDistance_y(Length_y*Radius_y_tri[i], Length_y*Radius_y_tri[j]);
    double distZ = Radius_z_tri[i] - Radius_z_tri[j];
    return pow(pow(distX,2.0) + pow(distY,2.0) + pow(distZ,2.0),0.5);
}
				
// Steps to compute energy of system from nothing
// Sum over all vertices, perform algorithm
// Need function to find angles opposite a link to compute theta_1, theta_2 to get
// sigma_ij = l_ij (cot(theta_1)+cot(theta_2))/2
// Compute these first
// Then compute sigma_i
// Then compute energy
// Interactions depend only on nodes linked to, hence when a node is changed we recompute energy at
// Node changed, nodes connected to node change as one sigma_ij component changes, thus have to recompute everything

void generateNeighborList() {    
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

// From here on out, will update the neighbor list everystep for each particle move/actin growth

void areaNode(int i) {
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

void normalTriangle(int i, double normal[3]) {
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

void shuffle_saru(Saru& saru, vector<int> &vector_int) {
    for(int i=(vector_int.size()-1); i>0; i--) {
        swap(vector_int[i], vector_int[saru.rand_select(i)]);
    }
}

void generateCheckerboard() {    
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

double cosineAngle(int i, int j, int k) {
    // Compute angle given by ij, ik
    double x1 = wrapDistance_x(Length_x*Radius_x_tri[j], Length_x*Radius_x_tri[i]);
    double y1 = wrapDistance_y(Length_y*Radius_y_tri[j], Length_y*Radius_y_tri[i]);
    double z1 = Radius_z_tri[j] - Radius_z_tri[i];
    double x2 = wrapDistance_x(Length_x*Radius_x_tri[k], Length_x*Radius_x_tri[i]);
    double y2 = wrapDistance_y(Length_y*Radius_y_tri[k], Length_y*Radius_y_tri[i]);
    double z2 = Radius_z_tri[k] - Radius_z_tri[i];
    return acos_fast((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
}

double cosineAngle_norm(int i, int j, int k) {
    // Compute angle given by ij, ik
    double x1 = wrapDistance_x(Length_x*Radius_x_tri[j], Length_x*Radius_x_tri[i]);
    double y1 = wrapDistance_y(Length_y*Radius_y_tri[j], Length_y*Radius_y_tri[i]);
    double z1 = Radius_z_tri[j] - Radius_z_tri[i];
    double x2 = wrapDistance_x(Length_x*Radius_x_tri[k], Length_x*Radius_x_tri[i]);
    double y2 = wrapDistance_y(Length_y*Radius_y_tri[k], Length_y*Radius_y_tri[i]);
    double z2 = Radius_z_tri[k] - Radius_z_tri[i];
    return acos((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
}

double cotangent(double x) {
//    return cos(x)/sin(x);
    return tan(M_PI_2 - x);
}

void acos_fast_initialize() {    
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

inline double acos_fast(double x) {
    double look_up = (x-acos_lu_min)*in_lu_interval_acos;
    int look_up_value = floor(look_up);
    return lu_values_acos[look_up_value]+(lu_values_acos[look_up_value+1]-lu_values_acos[look_up_value])*(look_up-look_up_value);
}

void cotangent_fast_initialize() {
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

inline double cotangent_fast(double x) {
    double look_up = (x-cot_lu_min)*in_lu_interval_cot;
    int look_up_value = floor(look_up);
    return lu_values_cot[look_up_value]+(lu_values_cot[look_up_value+1]-lu_values_cot[look_up_value])*(look_up-look_up_value);
}

void linkMaxMin() {
    cout.rdbuf(myfilebuf);
	double min = pow(10,9);
	double max = -1;
	cout << "Link lengths initially" << endl;
    #pragma omp parallel for reduction(max : max) reduction(min : min)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            double link_length = lengthLink(i,point_neighbor_list[i][j]);
            // cout << i << " " << j << " " << link_length << endl;
            /*
            if (link_length < 1.4) {
                cout << "Too low!" << endl;
                cout << "i " << Radius_x_tri[i] << " " << Radius_y_tri[i] << " " << Radius_z_tri[i] << endl;
                cout << "j " << Radius_x_tri[j] << " " << Radius_y_tri[j] << " " << Radius_z_tri[j] << endl;
            }
            if (link_length > 1.6) {
                cout << "Too high!" << endl;
                cout << "i " << Radius_x_tri[i] << " " << Radius_y_tri[i] << " " << Radius_z_tri[i] << endl;
                cout << "j " << Radius_x_tri[j] << " " << Radius_y_tri[j] << " " << Radius_z_tri[j] << endl;
            }
            */
            if(link_length > max) {
                max = link_length;
            } 
            if(link_length < min) {
                min = link_length;
            } 
        }
    } 
    // Let's change this to using the neighbor list.......
    #pragma omp parallel for reduction(min : min)
    for(int l=0; l<vertices; l++) {
        int index = neighbor_list_index[l];
	    for(int i=0; i<neighbors[index].size(); i++) {
            for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
                if(l != neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = lengthLink(l,neighbor_list[neighbors[index][i]][j]);
                    if(length_neighbor < min) {
                        min = length_neighbor;
                    } 
                }
            }
        }
    } 
	cout << "Min is " << min << "\n";
	cout << "Max is " << max << "\n";
}

void energyNode(int i) {
    // Compute energy about a node
	phi_vertex[i] = 0;
    double link_length[point_neighbor_max[i]];
    double opposite_angles[point_neighbor_max[i]][2];
    double sigma_i = 0;
    double sigma_ij[point_neighbor_max[i]];
    double energy_return_x = 0;
    double energy_return_y = 0;
    double energy_return_z = 0;
    // Compute link lengths for neighbor list
    for(int j=0; j<point_neighbor_max[i]; j++) {
        int k = point_neighbor_list[i][j];
        link_length[j] = lengthLink(i,k);
        if ((link_length[j] > 1.673) || (link_length[j] < 1.00)) {
            phi_vertex[i] = pow(10,100);
			// cout << "Breaks constraints at " << i << " " << k << endl;
			// cout << link_length[j] << endl;
            return;    
        }
        // cout << "Link length at " << i << " " << k << " " << " is " << link_length[j] << endl;
    }
    // Compute angles opposite links
    for(int j_1=0; j_1<point_neighbor_max[i]; j_1++) {
        int face_1 = point_neighbor_triangle[i][j_1][0];
        int face_2 = point_neighbor_triangle[i][j_1][1];
        // cout << "Points are for triangle are " << i << " " << point_neighbor_list[i][j_1] << endl;
        // cout << "Triangle 1 " << triangle_list[face_1][0] << " " << triangle_list[face_1][1] << " " << triangle_list[face_1][2] << endl;
        // cout << "Triangle 2 " << triangle_list[face_2][0] << " " << triangle_list[face_2][1] << " " << triangle_list[face_2][2] << endl;
        // From given faces, determine point in triangles not given by i or neighbor list 
        // Then use cosineAngle to get the angles, with opposite being the location opposite
        if((triangle_list[face_1][0] != i) && (triangle_list[face_1][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][0];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_1][1] != i) && (triangle_list[face_1][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][1];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_1][2];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }

        if((triangle_list[face_2][0] != i) && (triangle_list[face_2][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][0];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_2][1] != i) && (triangle_list[face_2][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][1];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_2][2];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
		/*
        if((opposite_angles[j_1][1] > M_PI_2*1.18) || (opposite_angles[j_1][0] > M_PI_2*1.18)) {
           phi_vertex[i] = 1000000;
           return;
        }
		*/
    }

    // Compute sigma_ij
    for(int j=0; j<point_neighbor_max[i]; j++) {
        // cout << "Opposite angles at " << i << " are " << opposite_angles[j][0] << " and " << opposite_angles[j][1] << endl;
        sigma_ij[j] = link_length[j]*(cotangent_fast(opposite_angles[j][0])+cotangent_fast(opposite_angles[j][1]))*0.5;
        // cout << "Sigma_ij at " << j << " is " << sigma_ij[j] << endl;
        sigma_i += sigma_ij[j]*link_length[j];
    }
    sigma_i = sigma_i*0.25;
    // sigma_i_total += sigma_i;
    // cout << "sigma_i is " << sigma_i << endl;
    // Summation over neighbors for energy
    // cout << "Initial energy_return_z " << energy_return_z << endl;
    for(int j=0; j<point_neighbor_max[i]; j++) {
        double energy_constant = sigma_ij[j]/link_length[j];
        // cout << "Energy constant is " << energy_constant << endl;
        energy_return_x += energy_constant*wrapDistance_x(Length_x*Radius_x_tri[i], Length_x*Radius_x_tri[point_neighbor_list[i][j]]);
        energy_return_y += energy_constant*wrapDistance_y(Length_y*Radius_y_tri[i], Length_y*Radius_y_tri[point_neighbor_list[i][j]]);
        energy_return_z += energy_constant*(Radius_z_tri[i] - Radius_z_tri[point_neighbor_list[i][j]]);
        // cout << "Z different is " << Radius_z_tri[i] << " minus " << Radius_z_tri[point_neighbor_list[i][j]] << " equals " << energy_return_z << endl;
    }
    // cout << "Energy values are x: " << energy_return_x << " y: " << energy_return_y << " z: " << energy_return_z << endl;
    // Don't ask me why but this comment makes the values of phi stable
    // I'm serious
    // cout << "Energy at vertex " << i << " : " << phi_vertex[i] << endl;
    // Calculate mean curvature if a protein node
    // get vertex normal
    // Evaluating vertex normal by taking average weighted by triangle area
    // Energy calculation structured to have area update before this part
    double vertex_normal[3] = {0,0,0};
    double vertex_area = 0;
    for(int j=0; j<point_triangle_max[i]; j++) {
        vertex_area += area_faces[point_triangle_list[i][j]];
    }
    for(int j=0; j<point_triangle_max[i]; j++) {
        double triangle_normal[3] = {0,0,0};
        normalTriangle(point_triangle_list[i][j], triangle_normal);
        vertex_normal[0] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[0];
        vertex_normal[1] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[1];
        vertex_normal[2] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[2];
    }
    // Now make sure vector is normalized
    double magnitude = pow(pow(vertex_normal[0],2.0)+pow(vertex_normal[1],2.0)+pow(vertex_normal[2],2.0),0.5);
    vertex_normal[0] = vertex_normal[0]/magnitude;
    vertex_normal[1] = vertex_normal[1]/magnitude;
    vertex_normal[2] = vertex_normal[2]/magnitude;
    // Now can get mean curvature
    mean_curvature_vertex[i] = 1.0/sigma_i*(vertex_normal[0]*energy_return_x+vertex_normal[1]*energy_return_y+vertex_normal[2]*energy_return_z);
    sigma_vertex[i] = sigma_i;
    double diff_curv = mean_curvature_vertex[i]-spon_curv[Ising_Array[i]];
    phi_vertex[i] = k_b[Ising_Array[i]]*sigma_i*diff_curv*diff_curv;
}

void initializeEnergy() {
    Phi = 0;
    Area_total = 0;
    // Loop through neighbor list to see if any hard sphere constraints are violated
	// Check to make sure not counting self case
    #pragma omp parallel for reduction(+:Phi)
    for(int k=0; k<vertices; k++) {
        int index = neighbor_list_index[k];
        for(int i=0; i<neighbors[index].size(); i++) {
            for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
                // Check particle interactions
                if(k != neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = lengthLink(k,neighbor_list[neighbors[index][i]][j]);
                    if(length_neighbor < 1.0) {
                        Phi += pow(10,100);
                    }
                }
            }
        }
    }
    // Compute surface area
    #pragma omp parallel for reduction(+:Area_total,Phi)
    for(int i=0; i<faces; i++) {
        areaNode(i);
        area_faces_original[i] = area_faces[i];
        Area_total += area_faces[i];
        Phi += gamma_surf[0]*area_faces[i]; // On second though, gamma_surf here might be as easy as assigning a value from Ising_Array as it is associated with a face value rather than a vertix. Maybe best to take average of vertices on face
    }
    Phi -= tau_frame*Length_x*Length_y;

    Phi_bending = 0;
    #pragma omp parallel for reduction(+:Phi_bending)
    for(int i=0; i<vertices; i++) {
        energyNode(i); // Contribution due to mean curvature and surface area
        phi_vertex_original[i] = phi_vertex[i];
        mean_curvature_vertex_original[i] = mean_curvature_vertex[i];
        sigma_vertex_original[i] = sigma_vertex[i];
        Phi_bending += phi_vertex[i];
    }
    Phi += Phi_bending;

    // Evaluate Ising model like energy
    Mass = 0;
    Magnet = 0;

    #pragma omp parallel for reduction(+:Mass,Magnet)
    for(int i=0; i<vertices; i++) {
        if(Ising_Array[i] < 2) {
            Mass += Ising_Array[i];
            Magnet += ising_values[Ising_Array[i]];
        }
    }
    Phi -= h_external*Magnet;   
 
    double Phi_magnet = 0;
    #pragma omp parallel for reduction(+:Phi_magnet)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            Phi_magnet -= J_coupling[Ising_Array[i]][Ising_Array[point_neighbor_list[i][j]]]*ising_values[Ising_Array[i]]*ising_values[Ising_Array[point_neighbor_list[i][j]]];
        }
    }
    Phi += Phi_magnet*0.5; // Dividing by 2 as double counting
}

void initializeEnergy_scale() {
    Phi = 0;
    Area_total = 0;
    // Loop through neighbor list to see if any hard sphere constraints are violated
	// Check to make sure not counting self case
    #pragma omp parallel for reduction(+:Phi)
    for(int k=0; k<vertices; k++) {
        int index = neighbor_list_index[k];
        for(int i=0; i<neighbors[index].size(); i++) {
            for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
                // Check particle interactions
                if(k != neighbor_list[neighbors[index][i]][j]) {
                    double length_neighbor = lengthLink(k,neighbor_list[neighbors[index][i]][j]);
                    if(length_neighbor < 1.0) {
                        Phi += pow(10,100);
                    }
                }
            }
        }
    }

    // If condition violated, Phi > 10^100 so we can just return
    if(Phi > pow(10,10)) {
        return;
    }
    // Compute surface area
    #pragma omp parallel for
    for(int i=0; i<faces; i++) {
        areaNode(i);
    }

    #pragma omp parallel for reduction(+:Area_total,Phi)
    for(int i=0; i<faces; i++) {
        Area_total += area_faces[i];
        Phi += gamma_surf[0]*area_faces[i]; // On second though, gamma_surf here might be as easy as assigning a value from Ising_Array as it is associated with a face value rather than a vertix. Maybe best to take average of vertices on face
    }
    Phi -= tau_frame*Length_x*Length_y;

    #pragma omp parallel for
    for(int i=0; i<vertices; i++) {
        energyNode(i); // Contribution due to mean curvature and surface area
    }

    // Idea is to seperate energy evaluation and adding Phi to allow for sweet vectorization
    Phi_bending = 0.0;
    #pragma omp parallel for reduction(+:Phi_bending)
    for(int i=0; i<vertices; i++) {
        Phi_bending += phi_vertex[i];
    }
    Phi += Phi_bending;

    // Evaluate Ising model like energy
    Phi -= h_external*Magnet;   
 
    double Phi_magnet = 0;
    #pragma omp parallel for reduction(+:Phi_magnet)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            Phi_magnet -= J_coupling[Ising_Array[i]][Ising_Array[point_neighbor_list[i][j]]]*ising_values[Ising_Array[i]]*ising_values[Ising_Array[point_neighbor_list[i][j]]];
        }
    }
    Phi += Phi_magnet*0.5; // Dividing by 2 as double counting
    Phi_phi = Phi_magnet*0.5;
}

void energyNode_i(int i) {
    // Compute energy about a node
	phi_vertex[i] = 0;
    double link_length[point_neighbor_max[i]];
    double opposite_angles[point_neighbor_max[i]][2];
    double sigma_i = 0;
    double sigma_ij[point_neighbor_max[i]];
    double energy_return_x = 0;
    double energy_return_y = 0;
    double energy_return_z = 0;
    // Compute link lengths for neighbor list
    for(int j=0; j<point_neighbor_max[i]; j++) {
        int k = point_neighbor_list[i][j];
        link_length[j] = lengthLink(i,k);
        if ((link_length[j] > 1.673) || (link_length[j] < 1.00)) {
            phi_vertex[i] = pow(10,100);
			// cout << "Breaks constraints at " << i << " " << k << endl;
            // return;    
        }
        // cout << "Link length at " << i << " " << k << " " << " is " << link_length[j] << endl;
    }
    // Compute angles opposite links
    for(int j_1=0; j_1<point_neighbor_max[i]; j_1++) {
        int face_1 = 0;
        int face_2 = 0;
        face_1 = point_neighbor_triangle[i][j_1][0];
        face_2 = point_neighbor_triangle[i][j_1][1];
        // cout << "Points are for triangle are " << i << " " << point_neighbor_list[i][j_1] << endl;
        // cout << "Triangle 1 " << triangle_list[face_1][0] << " " << triangle_list[face_1][1] << " " << triangle_list[face_1][2] << endl;
        // cout << "Triangle 2 " << triangle_list[face_2][0] << " " << triangle_list[face_2][1] << " " << triangle_list[face_2][2] << endl;
        // From given faces, determine point in triangles not given by i or neighbor list 
        // Then use cosineAngle to get the angles, with opposite being the location opposite
        if((triangle_list[face_1][0] != i) && (triangle_list[face_1][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][0];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_1][1] != i) && (triangle_list[face_1][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_1][1];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_1][2];
            opposite_angles[j_1][0] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }

        if((triangle_list[face_2][0] != i) && (triangle_list[face_2][0] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][0];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else if((triangle_list[face_2][1] != i) && (triangle_list[face_2][1] != point_neighbor_list[i][j_1])) {
            int opposite = triangle_list[face_2][1];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
        else {
            int opposite = triangle_list[face_2][2];
            opposite_angles[j_1][1] = cosineAngle(opposite, i, point_neighbor_list[i][j_1]);
        }
		/*
        if((opposite_angles[j_1][1] > M_PI_2*1.18) || (opposite_angles[j_1][0] > M_PI_2*1.18)) {
           phi_vertex[i] = 1000000;
           return;
        }
		*/
    }

    // Compute sigma_ij
    for(int j=0; j<point_neighbor_max[i]; j++) {
        // cout << "Opposite angles at " << i << " are " << opposite_angles[j][0] << " and " << opposite_angles[j][1] << endl;
        sigma_ij[j] = link_length[j]*(cotangent_fast(opposite_angles[j][0])+cotangent_fast(opposite_angles[j][1]))/2.0;
        // cout << "Sigma_ij at " << j << " is " << sigma_ij[j] << endl;
        sigma_i += sigma_ij[j]*link_length[j];
    }
    sigma_i = sigma_i/4.0;
    #pragma omp atomic
    sigma_i_total += sigma_i;
    // cout << "sigma_i is " << sigma_i << endl;
    // Summation over neighbors for energy
    // cout << "Initial energy_return_z " << energy_return_z << endl;
    for(int j=0; j<point_neighbor_max[i]; j++) {
        double energy_constant = sigma_ij[j]/link_length[j];
        // cout << "Energy constant is " << energy_constant << endl;
        energy_return_x += energy_constant*wrapDistance_x(Length_x*Radius_x_tri[i], Length_x*Radius_x_tri[point_neighbor_list[i][j]]);
        energy_return_y += energy_constant*wrapDistance_y(Length_y*Radius_y_tri[i], Length_y*Radius_y_tri[point_neighbor_list[i][j]]);
        energy_return_z += energy_constant*(Radius_z_tri[i] - Radius_z_tri[point_neighbor_list[i][j]]);
        // cout << "Z different is " << Radius_z_tri[i] << " minus " << Radius_z_tri[point_neighbor_list[i][j]] << " equals " << energy_return_z << endl;
    }
    // cout << "Energy values are x: " << energy_return_x << " y: " << energy_return_y << " z: " << energy_return_z << endl;
    double vertex_normal[3] = {0,0,0};
    double vertex_area = 0;
    for(int j=0; j<point_triangle_max[i]; j++) {
        vertex_area += area_faces[point_triangle_list[i][j]];
    }
    for(int j=0; j<point_triangle_max[i]; j++) {
        double triangle_normal[3] = {0,0,0};
        normalTriangle(point_triangle_list[i][j], triangle_normal);
        vertex_normal[0] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[0];
        vertex_normal[1] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[1];
        vertex_normal[2] += area_faces[point_triangle_list[i][j]]/vertex_area*triangle_normal[2];
    }
    // Now make sure vector is normalized
    double magnitude = pow(pow(vertex_normal[0],2.0)+pow(vertex_normal[1],2.0)+pow(vertex_normal[2],2.0),0.5);
    vertex_normal[0] = vertex_normal[0]/magnitude;
    vertex_normal[1] = vertex_normal[1]/magnitude;
    vertex_normal[2] = vertex_normal[2]/magnitude;
    // Now can get mean curvature
    mean_curvature_vertex[i] = 1.0/sigma_i*(vertex_normal[0]*energy_return_x+vertex_normal[1]*energy_return_y+vertex_normal[2]*energy_return_z);
    sigma_vertex[i] = sigma_i;
    double diff_curv = mean_curvature_vertex[i]-spon_curv[Ising_Array[i]];
    phi_vertex[i] = k_b[Ising_Array[i]]*sigma_i*diff_curv*diff_curv;
}


void initializeEnergy_i() {
    Phi = 0;
    Area_total = 0;
    // Compute surface area
    #pragma omp parallel for reduction(+:Area_total,Phi)
    for(int i=0; i<faces; i++) {
        areaNode(i);
        area_faces_original[i] = area_faces[i];
        Area_total += area_faces[i];
        Phi += gamma_surf[0]*area_faces[i]; // On second though, gamma_surf here might be as easy as assigning a value from Ising_Array as it is associated with a face value rather than a vertix. Maybe best to take average of vertices on face
    }
    Phi -= tau_frame*Length_x*Length_y;

    Phi_bending = 0.0;
    #pragma omp parallel for reduction(+:Phi_bending)
    for(int i=0; i<vertices; i++) {
        energyNode_i(i); // Contribution due to mean curvature and surface area
        phi_vertex_original[i] = phi_vertex[i];
        Phi_bending += phi_vertex[i];
    }
    Phi += Phi_bending;

    // Evaluate Ising model like energy
    Mass = 0;
    Magnet = 0;
    #pragma omp parallel for reduction(+:Mass,Magnet)
    for(int i=0; i<vertices; i++) {
        if(Ising_Array[i] < 2) {
            Mass += Ising_Array[i];
            Magnet += ising_values[Ising_Array[i]];
        }
    }
    Phi -= h_external*Magnet;   
 
    double Phi_magnet = 0;
    #pragma omp parallel for reduction(+:Phi_magnet)
    for(int i=0; i<vertices; i++) {
        for(int j=0; j<point_neighbor_max[i]; j++) {
            Phi_magnet -= J_coupling[Ising_Array[i]][Ising_Array[point_neighbor_list[i][j]]]*ising_values[Ising_Array[i]]*ising_values[Ising_Array[point_neighbor_list[i][j]]];
        }
    }
    Phi += Phi_magnet*0.5; // Dividing by 2 as double counting
    Phi_phi = Phi_magnet*0.5;
}

void DisplaceStep(int vertex_trial, int thread_id) {
    // Pick random site and translate node
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
	// cout << "Vertex trial " << vertex_trial << endl;
	// cout << "x " << Radius_x_tri[vertex_trial] << " y " << Radius_y_tri[vertex_trial] << " z " << Radius_z_tri[vertex_trial] << endl;
    // Trial move - change radius
    // Comment x,y out for z moves only
    Radius_x_tri[vertex_trial] += lambda*local_generator.d(-1.0,1.0);
    Radius_y_tri[vertex_trial] += lambda*local_generator.d(-1.0,1.0);
    Radius_z_tri[vertex_trial] += lambda*Length_y*local_generator.d(-1.0,1.0);
    // Apply PBC on particles
    Radius_x_tri[vertex_trial] = fmod(fmod(Radius_x_tri[vertex_trial],1.0)+1.0,1.0);
    Radius_y_tri[vertex_trial] = fmod(fmod(Radius_y_tri[vertex_trial],1.0)+1.0,1.0);
	// cout << "x " << Radius_x_tri[vertex_trial] << " y " << Radius_y_tri[vertex_trial] << " z " << Radius_z_tri[vertex_trial] << endl;
    double Phi_diff = 0; 
    double Phi_diff_bending = 0;
	// Loop through neighbor lists
	// Compare versus looping through all for verification
	// Determine index of current location
	int index_x = int(Length_x*Radius_x_tri[vertex_trial]/box_x);
	int index_y = int(Length_y*Radius_y_tri[vertex_trial]/box_y);
	int index_z = int((Radius_z_tri[vertex_trial]+Length_z)/box_z);
    if(index_x == nl_x) {
        index_x -= 1;
    }
    if(index_y == nl_y) {
        index_y -= 1;
    }
	int index = index_x + index_y*nl_x + index_z*nl_x*nl_y;
	// Loop through neighboring boxes
	// Check to make sure not counting self case
	/*
	cout << "At neighbor part!" << endl;
	cout << "Vertex trial" << endl;
	cout << "Index " << index << endl;
	*/
	// cout << "Displace step" << endl;
	for(int i=0; i<neighbors[index].size(); i++) {
		// cout << neighbors[index][i] << " ";
		// cout << "Number of particles in bin " << neighbor_list[neighbors[index][i]].size() << endl;
		// cout << "Particles time" << endl;
		for(int j=0; j<neighbor_list[neighbors[index][i]].size(); j++) {
			// cout << neighbor_list[neighbors[index][i]][j] << " ";
			// Check particle interactions
            if(vertex_trial != neighbor_list[neighbors[index][i]][j]) {
                double length_neighbor = lengthLink(vertex_trial,neighbor_list[neighbors[index][i]][j]);
                // cout << "length from particle to particle " << length_neighbor << endl;
                if(length_neighbor < 1.0) {
                    // cout << "Neighbor " << neighbor_list[neighbors[index][i]][j] << " " << Radius_x_tri[neighbor_list[neighbors[index][i]][j]] << " " << Radius_y_tri[neighbor_list[neighbors[index][i]][j]] << " " << Radius_z_tri[neighbor_list[neighbors[index][i]][j]] << endl;
                    Phi_diff += pow(10,100);
                }
            }
		}
		// cout << endl;
	}
	// cout << "Displace check " << check << endl;
	// cout << "Phi_diff after neighbor " << Phi_diff << endl;
	// cout << "Probability " << exp(-Phi_diff/T) << endl;
    // Check to see if particle moved out of bound
    int index_checkerboard_x = floor((Length_x*Radius_x_tri[vertex_trial]-cell_center_x)/box_x_checkerboard);
	int index_checkerboard_y = floor((Length_y*Radius_y_tri[vertex_trial]-cell_center_y)/box_y_checkerboard);
    // cout << index_checkerboard_x << " " << index_checkerboard_y << endl;
    if(index_checkerboard_x == -1) {
        index_checkerboard_x += checkerboard_x;
    }
    if(index_checkerboard_y == -1) {
        index_checkerboard_y += checkerboard_y;
    }
    if(index_checkerboard_x == checkerboard_x) {
        index_checkerboard_x -= 1;
    }
    if(index_checkerboard_y == checkerboard_y) {
        index_checkerboard_y -= 1;
    }
    // cout << "After " << index_checkerboard_x << " " << index_checkerboard_y << endl;
    int index_checkerboard = index_checkerboard_x + index_checkerboard_y*checkerboard_x;
    if(index_checkerboard != checkerboard_index[vertex_trial]) {
        Phi_diff += pow(10,100);
    }
    
    // Energy due to mean curvature
	/*
    int max_local_neighbor = 0;
    for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
        max_local_neighbor += point_neighbor_max[point_neighbor_list[vertex_trial][i]];
    }
    int local_neighbor_list[max_local_neighbor];
    int counter = 0;
    for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
        for(int j=0; j<point_neighbor_max[point_neighbor_list[vertex_trial][i]]; j++) {
            local_neighbor_list[counter] = point_neighbor_list[point_neighbor_list[vertex_trial][i]][j];
            counter++;
        }     
    }
    // /
    for(int i=0; i<max_local_neighbor; i++) {
        cout << local_neighbor_list[i] << " ";
    }
    cout << endl;
    // /
    sort(local_neighbor_list,local_neighbor_list+max_local_neighbor);
    // /
    for(int i=0; i<max_local_neighbor; i++) {
        cout << local_neighbor_list[i] << " ";
    }
    cout << endl;
    // / 
    int max_unique = 0;
    int neighbor_unique_raw[max_local_neighbor];
    // Go through sorted array
    for(int i=0; i<max_local_neighbor; i++) {
        while ((i < (max_local_neighbor-1)) && (local_neighbor_list[i] == local_neighbor_list[i+1])) {
            i++;
        }
        neighbor_unique_raw[max_unique] = local_neighbor_list[i]; 
        max_unique++;
    }
    // /
    for(int i=0; i<max_unique; i++) {
        cout << neighbor_unique_raw[i] << " ";
    }    
    cout << endl;
    // /
    for(int i=0; i<max_unique; i++) {
        energyNode(neighbor_unique_raw[i]);
        Phi_diff += phi_vertex[neighbor_unique_raw[i]] - phi_vertex_original[neighbor_unique_raw[i]];
    }
	*/
    bool accept = false;
    double Chance = local_generator.d();
    double Chance_factor = -T*log(Chance);
    if(Phi_diff < pow(10,10)) {
        // Evaluate energy difference
        // Energy due to surface area
        for(int i=0; i<point_triangle_max[vertex_trial]; i++) {
            int j = point_triangle_list[vertex_trial][i];
            areaNode(j);
            Phi_diff += gamma_surf[0]*(area_faces[j] - area_faces_original[j]);
        }
        // Evaluated at node changed and neighboring nodes
        energyNode(vertex_trial);
        Phi_diff_bending += phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            energyNode(point_neighbor_list[vertex_trial][i]);
            Phi_diff_bending += phi_vertex[point_neighbor_list[vertex_trial][i]] - phi_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        Phi_diff += Phi_diff_bending;
        // cout << "Phi_diff after phi_vertexes " << Phi_diff << endl;
        // cout << "Phi_diff after checking boundaries " << Phi_diff << endl;
    }
    /*
    for(int i=0; i<vertices; i++) {
        energyNode(i); // Contribution due to mean curvature and surface area
        Phi_diff += phi_vertex[i] - phi_vertex_original[i];
    }
    */

    // EnergyStep();
    // double Phi_diff = Phi - Phi_original;

    // Actual area
    /*
    double total_area = 0.0;
    for(int i=0; i<faces; i++) {
        areaNode(i);
        total_area += area_faces[i];
    }
    cout << "Actual area is " << total_area << endl;
    */
	// Check to see if displacement violates actin hard sphere
	/*
	if((abs(Radius_x_tri[vertex_trial]-Length_x/2.0) < (1.0/5.0*Length_x+1.0)) && (abs(Radius_y_tri[vertex_trial]-Length_y/2.0) < (1.0/4.0*Length_y+1.0))) {
		for(int i = 0; i < particle_coord_x.size(); i++) {
			double dist_check = pow(pow(Radius_x_tri[vertex_trial]-particle_coord_x[i],2)+pow(Radius_y_tri[vertex_trial]-particle_coord_y[i],2)+pow(Radius_z_tri[vertex_trial]-particle_coord_z[i],2),0.5);
			if (dist_check < 1.09) {
				// cout << i << " " << dist_check << endl;
				Phi_diff += pow(10,100);
			}
		}
	}
	*/
	// cout << "Phi_diff before neighbor list " << Phi_diff << endl;
    if((accept == true) || ((Chance_factor>Phi_diff) && (Phi_diff < pow(10,10)))) {
        // Accept move
        // New way that uses energy loop
        // Accept all trial values with for loop
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;

        /*
        for(int i=0; i<vertices; i++) {
            phi_vertex_original[i] = phi_vertex[i];
        }
        */
		/*
        for(int i=0; i<max_unique; i++) {
            phi_vertex_original[neighbor_unique_raw[i]] = phi_vertex[neighbor_unique_raw[i]];
        }  
        */
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            phi_vertex_original[point_neighbor_list[vertex_trial][i]] = phi_vertex[point_neighbor_list[vertex_trial][i]];
        }
        mean_curvature_vertex_original[vertex_trial] = mean_curvature_vertex[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            mean_curvature_vertex_original[point_neighbor_list[vertex_trial][i]] = mean_curvature_vertex[point_neighbor_list[vertex_trial][i]];
        }
        sigma_vertex_original[vertex_trial] = sigma_vertex[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            sigma_vertex_original[point_neighbor_list[vertex_trial][i]] = sigma_vertex[point_neighbor_list[vertex_trial][i]];
        }
        Radius_x_tri_original[vertex_trial] = Radius_x_tri[vertex_trial];        
        Radius_y_tri_original[vertex_trial] = Radius_y_tri[vertex_trial];
        Radius_z_tri_original[vertex_trial] = Radius_z_tri[vertex_trial];
        double Area_total_diff_local = 0.0;
        for(int i=0; i<point_triangle_max[vertex_trial]; i++) {
            int j = point_triangle_list[vertex_trial][i];
            Area_total_diff_local += area_faces[j] - area_faces_original[j];
            area_faces_original[j] = area_faces[j];
        }
        Area_diff_thread[thread_id][0] += Area_total_diff_local;
		// Change neighbor list if new index doesn't match up with old
		if(neighbor_list_index[vertex_trial] != index) {
			// Determine which entry vertex trial was in original index bin and delete
			// cout << "Deleting index!" << endl;
			// cout << "Old neighbor list and size " << neighbor_list[neighbor_list_index[vertex_trial]].size() <<  endl;
			/*
			for(int i=0; i<neighbor_list[neighbor_list_index[vertex_trial]].size(); i++) {
				cout << neighbor_list[neighbor_list_index[vertex_trial]][i] << " ";
			}
			*/
			// cout << endl;
			for(int i=0; i<neighbor_list[neighbor_list_index[vertex_trial]].size(); i++) {
				if(neighbor_list[neighbor_list_index[vertex_trial]][i] == vertex_trial) {
					neighbor_list[neighbor_list_index[vertex_trial]][i] = neighbor_list[neighbor_list_index[vertex_trial]].back();
					neighbor_list[neighbor_list_index[vertex_trial]].pop_back();
					i += neighbor_list[neighbor_list_index[vertex_trial]].size()+10;
				}
			}
			// cout << "Should have been deleted and size " << neighbor_list[neighbor_list_index[vertex_trial]].size() << endl;
			/*
			for(int i=0; i<neighbor_list[neighbor_list_index[vertex_trial]].size(); i++) {
				cout << neighbor_list[neighbor_list_index[vertex_trial]][i] << " ";
			}
			*/
			// cout << endl;
			// Add to new bin
			// Old neighbor list
			// cout << "Old neighbor list at new index" << endl;
			/*
			for(int i=0; i<neighbor_list[index].size(); i++) {
				cout << neighbor_list[index][i] << " ";
			}
			*/
			// cout << endl;
			neighbor_list[index].push_back(vertex_trial);
			neighbor_list_index[vertex_trial] = index;
			// cout << "New neighbor list at new index" << endl;
			/*
			for(int i=0; i<neighbor_list[index].size(); i++) {
				cout << neighbor_list[index][i] << " ";
			}
			*/
			// cout << endl;
		}
    }
    else {
        steps_rejected_displace_thread[thread_id][0] += 1;
        Radius_x_tri[vertex_trial] = Radius_x_tri_original[vertex_trial];
        Radius_y_tri[vertex_trial] = Radius_y_tri_original[vertex_trial];
        Radius_z_tri[vertex_trial] = Radius_z_tri_original[vertex_trial];

        /*
        for(int i=0; i<vertices; i++) {
            phi_vertex[i] = phi_vertex_original[i];
        }
        */
		/*
        for(int i=0; i<max_unique; i++) {
            phi_vertex[neighbor_unique_raw[i]] = phi_vertex_original[neighbor_unique_raw[i]];
        } 
		*/     
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            phi_vertex[point_neighbor_list[vertex_trial][i]] = phi_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        mean_curvature_vertex[vertex_trial] = mean_curvature_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            mean_curvature_vertex[point_neighbor_list[vertex_trial][i]] = mean_curvature_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        sigma_vertex[vertex_trial] = sigma_vertex_original[vertex_trial];
        for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
            sigma_vertex[point_neighbor_list[vertex_trial][i]] = sigma_vertex_original[point_neighbor_list[vertex_trial][i]];
        }
        for(int i=0; i<point_triangle_max[vertex_trial]; i++) {
            int j = point_triangle_list[vertex_trial][i];
            area_faces[j] = area_faces_original[j];
        }
    }
    steps_tested_displace_thread[thread_id][0] += 1;
	// cout << "Displace step end" << endl;
}

void TetherCut(int vertex_trial, int thread_id) {
    // Choose link at random, destroy it, and create new link joining other ends of triangle
    // Have to update entries of triangle_list, point_neighbor_list, point_neighbor_triangle, point_triangle_list, and point_neighbor_max for this
    // Select random vertex
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
	// Reject move if acting on rollers
    // int vertex_trial = 1000;
    // Select random link from avaliable
    int link_trial = local_generator.rand_select(point_neighbor_max[vertex_trial]-1);
    int vertex_trial_opposite = point_neighbor_list[vertex_trial][link_trial];

	/*
    cout << "Begin tether" << endl;
    cout << "Vertex trial: " << vertex_trial << endl;
    cout << "Link trial: " << link_trial << endl;
    cout << "Vertex opposite: " << vertex_trial_opposite << endl;
	*/

    int triangle_trial[2]; 
    int point_trial[2]; 
    int point_trial_position[2];

    // Find the triangles to be changed using point_neighbor_list
    triangle_trial[0] = point_neighbor_triangle[vertex_trial][link_trial][0];
    triangle_trial[1] = point_neighbor_triangle[vertex_trial][link_trial][1];

	/*
    cout << "Triangle " << triangle_trial[0] << ": " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
    cout << "Triangle " << triangle_trial[1] << ": " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
	*/
    // Check to see if vertex, vertex opposite are even on the two triangles
	/*
    int check_vertex = 0;
    int check_vertex_op = 0;
    for(int i=0; i<3; i++) {
        if(triangle_list[triangle_trial[0]][i] == vertex_trial) {
            check_vertex++;
        }
        if(triangle_list[triangle_trial[0]][i] == vertex_trial_opposite) {
            check_vertex_op++;
        }
        if(triangle_list[triangle_trial[1]][i] == vertex_trial) {
            check_vertex++;
        }
        if(triangle_list[triangle_trial[1]][i] == vertex_trial_opposite) {
            check_vertex_op++;
        }
    }

    if((check_vertex != 2) || (check_vertex_op != 2)) {
        cout << "Broken triangles" << endl;
        cout << "End tether" << endl;
        return;
    }
	*/
    // Find the two other points in the triangles using faces
    for(int i=0; i<2; i++) {
        if ((vertex_trial == triangle_list[triangle_trial[i]][0]) || (vertex_trial_opposite == triangle_list[triangle_trial[i]][0])) {
            if ((vertex_trial == triangle_list[triangle_trial[i]][1]) || (vertex_trial_opposite == triangle_list[triangle_trial[i]][1])) {
                point_trial[i] = triangle_list[triangle_trial[i]][2];
                point_trial_position[i] = 2;
            }
            else {
                point_trial[i] = triangle_list[triangle_trial[i]][1];
                point_trial_position[i] = 1;
            }
        }
        else {
            point_trial[i] = triangle_list[triangle_trial[i]][0];
            point_trial_position[i] = 0;
        }
    }
	int triangle_break[2];
    if((vertex_trial < 0) || (vertex_trial_opposite < 0) || (point_trial[0] < 0) || (point_trial[1] < 0) || (vertex_trial_opposite > vertices) || (point_trial[0] > vertices) || (point_trial[1] > vertices)) {
        /*
        cout << "It's breaking " << vertex_trial << " " << vertex_trial_opposite << " " << point_trial[0] << " " << point_trial[1] << endl;
        cout << link_triangle_list[point_trial[0]][point_trial[1]][0] << " " << link_triangle_list[point_trial[0]][point_trial[1]][1] << " " << link_triangle_list[point_trial[1]][point_trial[0]][0] << " " << link_triangle_list[point_trial[1]][point_trial[0]][1] << endl;
		cout << "Triangle " << triangle_trial[0] << " : " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
		cout << "Triangle " << triangle_trial[1] << " : " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
		cout << "Triangles that are breaking" << endl;
		if ((link_triangle_list[point_trial[0]][point_trial[1]][0] > 0) || (link_triangle_list[point_trial[0]][point_trial[1]][1] > 0)) {
			triangle_break[0] = link_triangle_list[point_trial[0]][point_trial[1]][0];
			triangle_break[1] = link_triangle_list[point_trial[0]][point_trial[1]][1];
			cout << "Triangle " << triangle_break[0] << " : " << triangle_list[triangle_break[0]][0] << " " << triangle_list[triangle_break[0]][1] << " " << triangle_list[triangle_break[0]][2] << endl;
			cout << "Triangle " << triangle_break[1] << " : " << triangle_list[triangle_break[1]][0] << " " << triangle_list[triangle_break[1]][1] << " " << triangle_list[triangle_break[1]][2] << endl;
		}
		if ((link_triangle_list[point_trial[1]][point_trial[1]][0] > 0) || (link_triangle_list[point_trial[1]][point_trial[0]][1] > 0)) {
			triangle_break[0] = link_triangle_list[point_trial[1]][point_trial[0]][0];
			triangle_break[1] = link_triangle_list[point_trial[1]][point_trial[0]][1];
			cout << "Triangle " << triangle_break[0] << " : " << triangle_list[triangle_break[0]][0] << " " << triangle_list[triangle_break[0]][1] << " " << triangle_list[triangle_break[0]][2] << endl;
			cout << "Triangle " << triangle_break[1] << " : " << triangle_list[triangle_break[1]][0] << " " << triangle_list[triangle_break[1]][1] << " " << triangle_list[triangle_break[1]][2] << endl;
		}
		cout << "Link trial " << link_trial << " neighbor max " << point_neighbor_max[vertex_trial] << endl;
		dumpLAMMPSConfig("lconfig_break");
        cout << "End tether" << endl;
        */
		steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if point_trial[0] and point_trial[1] are already linked
    for(int i=0; i<point_neighbor_max[point_trial[0]]; i++) {
        if(point_neighbor_list[point_trial[0]][i] == point_trial[1]) {
            steps_rejected_tether_thread[thread_id][0] += 1;
            steps_tested_tether_thread[thread_id][0] += 1;
            return;
        }
    }
    
    // cout << "Other points are: " << point_trial[0] << " " << point_trial[1] << " at " << point_trial_position[0] << " " << point_trial_position[1] << endl;
	// Check to see if the limits for maximum or minimum number of neighbors is exceeded
	if(((point_neighbor_max[vertex_trial]-1) == neighbor_min) || ((point_neighbor_max[vertex_trial_opposite]-1) == neighbor_min) || ((point_neighbor_max[point_trial[0]]+1) == neighbor_max) || ((point_neighbor_max[point_trial[1]]+1) == neighbor_max)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        // cout << "End tether" << endl;
		return;
	}
    
    // Check to see if any of the points are outside of the checkerboard set that vertex_trial is in
    if((checkerboard_index[vertex_trial] != checkerboard_index[vertex_trial_opposite]) || (checkerboard_index[vertex_trial] != checkerboard_index[point_trial[0]]) || (checkerboard_index[vertex_trial] != checkerboard_index[point_trial[1]])) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        // cout << "End tether" << endl;
		return;
    }

    // Check to see if trial points are too far apart
    double distance_point_trial = lengthLink(point_trial[0], point_trial[1]); 
    if ((distance_point_trial > 1.673) || (distance_point_trial < 1.00)) {
        steps_rejected_tether_thread[thread_id][0] += 1;
    	steps_tested_tether_thread[thread_id][0] += 1;
        // cout << "End tether" << endl;
        return;    
    }

    // Calculate detailed balance factor before everything changes
    // Basically, we now acc(v -> w) = gen(w->v) P(w)/gen(v->w) P(v)
    // Probability of gen(v->w) is 1/N*(1/Neighbors at vertex trial + 1/Neighbors at vertex_trial_opposite)
    // Similar for gen(w->v) except with point trial
    double db_factor = (1.0/(double(point_neighbor_max[point_trial[0]])+1.0)+1.0/(double(point_neighbor_max[point_trial[1]])+1.0))/(1.0/double(point_neighbor_max[vertex_trial])+1.0/double(point_neighbor_max[vertex_trial_opposite]));

    // Have all points needed, now just time to change triangle_list, point_neighbor_list, point_neighbor_triangle, point_triangle_list, point_neighbor_max, and point_triangle_max
    // Change triangle_list
    // Check orientation to see if consistent with before
    for(int i=0; i<3; i++) {
        if(triangle_list[triangle_trial[0]][i] == vertex_trial_opposite) {
            triangle_list[triangle_trial[0]][i] = point_trial[1];
            break;
        }
    }
    for(int i=0; i<3; i++) {
        if(triangle_list[triangle_trial[1]][i] == vertex_trial) {
            triangle_list[triangle_trial[1]][i] = point_trial[0];
            break;
        }
    }

	/*
    cout << "Triangle " << triangle_trial[0] << ": " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
    cout << "Triangle " << triangle_trial[1] << ": " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
	*/
    // Change point_neighbor_list and point_neighbor_triangle
    // Delete points
    int placeholder_nl = 0;
    int placeholder_neighbor1 = 0;
    int placeholder_neighbor2 = 0;
    // cout << "Neighbor list for point " << vertex_trial << " :";
    while(placeholder_nl < point_neighbor_max[vertex_trial]) {
        if(point_neighbor_list[vertex_trial][placeholder_nl] == vertex_trial_opposite) {
            // cout << " " << point_neighbor_list[vertex_trial][placeholder_nl] << " ";
            point_neighbor_list[vertex_trial][placeholder_nl] = point_neighbor_list[vertex_trial][point_neighbor_max[vertex_trial]-1];
            point_neighbor_triangle[vertex_trial][placeholder_nl][0] = point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]-1][0];
            point_neighbor_triangle[vertex_trial][placeholder_nl][1] = point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]-1][1];
            placeholder_neighbor1 = placeholder_nl;
            point_neighbor_list[vertex_trial][point_neighbor_max[vertex_trial]-1] = -1;
            // cout << " now " << point_neighbor_list[vertex_trial][placeholder_nl];
            placeholder_nl = neighbor_max;
            // cout << " SKIP ";
        }
        // cout << " " << point_neighbor_list[vertex_trial][placeholder_nl] << " ";
        placeholder_nl += 1;
    }
    // cout << " end " << endl;
    point_neighbor_max[vertex_trial] -= 1;
    placeholder_nl = 0;
    while(placeholder_nl < point_neighbor_max[vertex_trial_opposite]) {
        if(point_neighbor_list[vertex_trial_opposite][placeholder_nl] == vertex_trial) {
            point_neighbor_list[vertex_trial_opposite][placeholder_nl] = point_neighbor_list[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1];
            point_neighbor_triangle[vertex_trial_opposite][placeholder_nl][0] = point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1][0];
            point_neighbor_triangle[vertex_trial_opposite][placeholder_nl][1] = point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1][1];
            placeholder_neighbor2 = placeholder_nl;
            point_neighbor_list[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]-1] = -1;
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    point_neighbor_max[vertex_trial_opposite] -= 1;
    // Add points
    point_neighbor_list[point_trial[0]][point_neighbor_max[point_trial[0]]] = point_trial[1];
    point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]][0] = triangle_trial[0];
    point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]][1] = triangle_trial[1];
    point_neighbor_list[point_trial[1]][point_neighbor_max[point_trial[1]]] = point_trial[0];
    point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]][0] = triangle_trial[0];
    point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]][1] = triangle_trial[1];
    point_neighbor_max[point_trial[0]] += 1;
    point_neighbor_max[point_trial[1]] += 1;

    // Note that the definition of triangle_trial[0] and triangle_trial[1] have changed
    // Need to modify point_neighbor_triangle entries between vertex_trial and point_trial[1]
    // and vertex_trial_opposite and point_trial[0] to swap triangle_trial[1] to triangle_trial[0]
    // and triangle_trial[0] to triangle_trial[1] respectively
    // Placeholder values so places needed are saved
    int placeholder_remake[4] = {0,0,0,0};
    int placeholder_remake_01[4] = {0,0,0,0};
    // vertex_trial and point_trial[1]
    for(int i=0; i<point_neighbor_max[vertex_trial]; i++) {
        if(point_neighbor_list[vertex_trial][i] == point_trial[1]) {
            placeholder_remake[0] = i;
            if(point_neighbor_triangle[vertex_trial][i][0] == triangle_trial[1]) {
                point_neighbor_triangle[vertex_trial][i][0] = triangle_trial[0];
                placeholder_remake_01[0] = 0;
            }
            else if(point_neighbor_triangle[vertex_trial][i][1] == triangle_trial[1]) {
                point_neighbor_triangle[vertex_trial][i][1] = triangle_trial[0];
                placeholder_remake_01[0] = 1;
            }
        }
    }
    for(int i=0; i<point_neighbor_max[point_trial[1]]; i++) {
        if(point_neighbor_list[point_trial[1]][i] == vertex_trial) {
            placeholder_remake[1] = i;
            if(point_neighbor_triangle[point_trial[1]][i][0] == triangle_trial[1]) {
                point_neighbor_triangle[point_trial[1]][i][0] = triangle_trial[0];
                placeholder_remake_01[1] = 0;
            }
            else if(point_neighbor_triangle[point_trial[1]][i][1] == triangle_trial[1]) {
                point_neighbor_triangle[point_trial[1]][i][1] = triangle_trial[0];
                placeholder_remake_01[1] = 1;
            }
        }
    }
    // vertex_trial_opposite and point_trial[0]
    for(int i=0; i<point_neighbor_max[vertex_trial_opposite]; i++) {
        if(point_neighbor_list[vertex_trial_opposite][i] == point_trial[0]) {
            placeholder_remake[2] = i;
            if(point_neighbor_triangle[vertex_trial_opposite][i][0] == triangle_trial[0]) {
                point_neighbor_triangle[vertex_trial_opposite][i][0] = triangle_trial[1];
                placeholder_remake_01[2] = 0;
            }
            else if(point_neighbor_triangle[vertex_trial_opposite][i][1] == triangle_trial[0]) {
                point_neighbor_triangle[vertex_trial_opposite][i][1] = triangle_trial[1];
                placeholder_remake_01[2] = 1;
            }
        }
    }
    for(int i=0; i<point_neighbor_max[point_trial[0]]; i++) {
        if(point_neighbor_list[point_trial[0]][i] == vertex_trial_opposite) {
            placeholder_remake[3] = i;
            if(point_neighbor_triangle[point_trial[0]][i][0] == triangle_trial[0]) {
                point_neighbor_triangle[point_trial[0]][i][0] = triangle_trial[1];
                placeholder_remake_01[3] = 0;
            }
            else if(point_neighbor_triangle[point_trial[0]][i][1] == triangle_trial[0]) {
                point_neighbor_triangle[point_trial[0]][i][1] = triangle_trial[1];
                placeholder_remake_01[3] = 1;
            }
        }
    }

    // Change point_triangle_list
    placeholder_nl = 0;
    int placeholder_triangle1 = 0;
    int placeholder_triangle2 = 0;
    while(placeholder_nl < point_triangle_max[vertex_trial]) {
        if(point_triangle_list[vertex_trial][placeholder_nl] == triangle_trial[1]) {
            point_triangle_list[vertex_trial][placeholder_nl] = point_triangle_list[vertex_trial][point_triangle_max[vertex_trial]-1];
            placeholder_triangle1 = placeholder_nl;
            point_triangle_list[vertex_trial][point_triangle_max[vertex_trial]-1] = -1;
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    point_triangle_max[vertex_trial] -= 1;
    placeholder_nl = 0;
    while(placeholder_nl < point_triangle_max[vertex_trial_opposite]) {
        if(point_triangle_list[vertex_trial_opposite][placeholder_nl] == triangle_trial[0]) {
            point_triangle_list[vertex_trial_opposite][placeholder_nl] = point_triangle_list[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]-1];
            placeholder_triangle2 = placeholder_nl;
            point_triangle_list[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]-1] = -1;
            placeholder_nl = neighbor_max;
        }
        placeholder_nl += 1;
    }
    point_triangle_max[vertex_trial_opposite] -= 1;
    // Add points
    point_triangle_list[point_trial[0]][point_triangle_max[point_trial[0]]] = triangle_trial[1];
    point_triangle_list[point_trial[1]][point_triangle_max[point_trial[1]]] = triangle_trial[0];
    point_triangle_max[point_trial[0]] += 1;
    point_triangle_max[point_trial[1]] += 1;

    // Evaluate energy difference
    // Evaluated at four nodes from two triangles changed
    double Phi_diff = 0; 
    double Phi_diff_bending = 0;
    double Phi_diff_phi = 0;
    // Energy due to surface area
    areaNode(triangle_trial[0]);
    areaNode(triangle_trial[1]);
    Phi_diff += gamma_surf[0]*(area_faces[triangle_trial[0]] - area_faces_original[triangle_trial[0]]);
    // cout << "Phi_diff after " << vertex_trial << " " << Phi_diff << endl;
    Phi_diff += gamma_surf[0]*(area_faces[triangle_trial[1]] - area_faces_original[triangle_trial[1]]);
    // cout << "Phi_diff after " << vertex_trial << " " << Phi_diff << endl;
    // Energy due to mean curvature
    energyNode(vertex_trial);
    energyNode(vertex_trial_opposite);
    energyNode(point_trial[0]);
    energyNode(point_trial[1]);
    Phi_diff_bending += phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    // cout << "Phi_diff_bending after " << vertex_trial << " " << Phi_diff_bending << endl;
    Phi_diff_bending += phi_vertex[vertex_trial_opposite] - phi_vertex_original[vertex_trial_opposite];
    // cout << "Phi_diff_bending after " << vertex_trial_opposite << " " << Phi_diff_bending << endl;
    Phi_diff_bending += phi_vertex[point_trial[0]] - phi_vertex_original[point_trial[0]];
    // cout << "Phi_diff_bending after " << point_trial[0] << " " << Phi_diff_bending << endl;
    Phi_diff_bending += phi_vertex[point_trial[1]] - phi_vertex_original[point_trial[1]];
    // cout << "Phi_diff_bending after " << point_trial[1] << " " << Phi_diff_bending << endl;
    Phi_diff += Phi_diff_bending;

	Phi_diff_phi += J_coupling[Ising_Array[vertex_trial]][Ising_Array[vertex_trial_opposite]]*ising_values[Ising_Array[vertex_trial]]*ising_values[Ising_Array[vertex_trial_opposite]];
	Phi_diff_phi -= J_coupling[Ising_Array[point_trial[0]]][Ising_Array[point_trial[1]]]*ising_values[Ising_Array[point_trial[0]]]*ising_values[Ising_Array[point_trial[1]]];

    Phi_diff += Phi_diff_phi;
    // cout << "Phi_diff is " << Phi_diff << endl;
    double Chance = local_generator.d();
    double Chance_factor = -T*log(Chance/db_factor);
    bool accept = false;

	/*
	// Debugging
    double Phi_ = Phi + Phi_diff;
	cout << "Before init Phi " << " " << Phi << endl;
	initializeEnergy();
    if (Phi > pow(10,9)) {
		Phi -= 2*pow(10,9);
	}
	cout.precision(10);
    cout << "Phi " << Phi << " and Phi_ " << Phi_ << endl;
	cout << "Points " << vertex_trial << " " << vertex_trial_opposite << " " << point_trial[0] << " " << point_trial[1] << endl;
	cout << "Phi before chance in tether cut " << std::scientific << (Phi - Phi_) << endl;
	Phi = Phi_ - Phi_diff;
	*/

    if((accept == true) || ((Chance_factor>Phi_diff) && (Phi_diff < pow(10,10)))) {
        // Accept move
        // New way that uses energy loop
        // Accept all trial values with for loop
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;

        // Update original values
        // Have all points needed, now just time to change triangle_list, point_neighbor_list, link_triangle_list, point_triangle_list, point_neighbor_max, and point_triangle_max
        // Change triangle_list
        triangle_list_original[triangle_trial[0]][0] = triangle_list[triangle_trial[0]][0];
        triangle_list_original[triangle_trial[0]][1] = triangle_list[triangle_trial[0]][1];
        triangle_list_original[triangle_trial[0]][2] = triangle_list[triangle_trial[0]][2];

        triangle_list_original[triangle_trial[1]][0] = triangle_list[triangle_trial[1]][0];
        triangle_list_original[triangle_trial[1]][1] = triangle_list[triangle_trial[1]][1];
        triangle_list_original[triangle_trial[1]][2] = triangle_list[triangle_trial[1]][2];

		/*
        cout << "End accept " << vertex_trial << " " << vertex_trial_opposite << " " << point_trial[0] << " " << point_trial[1] << endl;
        cout << link_triangle_list_original[vertex_trial][vertex_trial_opposite][0] << " " << link_triangle_list_original[vertex_trial][vertex_trial_opposite][1] << " " << link_triangle_list_original[vertex_trial_opposite][vertex_trial][0] << " " << link_triangle_list_original[vertex_trial_opposite][vertex_trial][1] << endl;
        cout << link_triangle_list_original[point_trial[0]][point_trial[1]][0] << " " << link_triangle_list_original[point_trial[0]][point_trial[1]][1] << " " << link_triangle_list_original[point_trial[1]][point_trial[0]][0] << " " << link_triangle_list_original[point_trial[1]][point_trial[0]][1] << endl;
		cout << "Triangle " << triangle_trial[0] << " : " << triangle_list_original[triangle_trial[0]][0] << " " << triangle_list_original[triangle_trial[0]][1] << " " << triangle_list_original[triangle_trial[0]][2] << endl;
		cout << "Triangle " << triangle_trial[1] << " : " << triangle_list_original[triangle_trial[1]][0] << " " << triangle_list_original[triangle_trial[1]][1] << " " << triangle_list_original[triangle_trial[1]][2] << endl;
		*/
        // Check to see if point_trial[0], point_trial[1] are even on the two triangles
		/*
        int check_0 = 0;
        int check_1 = 0;
        for(int i=0; i<3; i++) {
            if(triangle_list_original[triangle_trial[0]][i] == point_trial[0]) {
                check_0++;
            }
            if(triangle_list_original[triangle_trial[0]][i] == point_trial[1]) {
                check_1++;
            }
            if(triangle_list_original[triangle_trial[1]][i] == point_trial[0]) {
                check_0++;
            }
            if(triangle_list_original[triangle_trial[1]][i] == point_trial[1]) {
                check_1++;
            }
        }

        if((check_0 != 2) || (check_1 != 2)) {
            cout << "Broken triangles in accept" << endl;
        }
		*/
        // Change point_neighbor_list
        // Delete points
        point_neighbor_list_original[vertex_trial][placeholder_neighbor1] = point_neighbor_list_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1];
        point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][0] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1][0];
        point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][1] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1][1];
        point_neighbor_list_original[vertex_trial][point_neighbor_max_original[vertex_trial]-1] = -1;
        point_neighbor_max_original[vertex_trial] -= 1;
        point_neighbor_list_original[vertex_trial_opposite][placeholder_neighbor2] = point_neighbor_list_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1];
        point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][0] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1][0];
        point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][1] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1][1];
        point_neighbor_list_original[vertex_trial_opposite][point_neighbor_max_original[vertex_trial_opposite]-1] = -1;
        point_neighbor_max_original[vertex_trial_opposite] -= 1;

        // Add points
        point_neighbor_list_original[point_trial[0]][point_neighbor_max_original[point_trial[0]]] = point_trial[1];
        point_neighbor_list_original[point_trial[1]][point_neighbor_max_original[point_trial[1]]] = point_trial[0];
        point_neighbor_triangle_original[point_trial[0]][point_neighbor_max_original[point_trial[0]]][0] = triangle_trial[0];
        point_neighbor_triangle_original[point_trial[0]][point_neighbor_max_original[point_trial[0]]][1] = triangle_trial[1];
        point_neighbor_triangle_original[point_trial[1]][point_neighbor_max_original[point_trial[1]]][0] = triangle_trial[0];
        point_neighbor_triangle_original[point_trial[1]][point_neighbor_max_original[point_trial[1]]][1] = triangle_trial[1];
        point_neighbor_max_original[point_trial[0]] += 1;
        point_neighbor_max_original[point_trial[1]] += 1;

        // Points changed due to redefinitions in triangle_trial[0] and triangle_trial[1]
        point_neighbor_triangle_original[vertex_trial][placeholder_remake[0]][placeholder_remake_01[0]] = triangle_trial[0];
        point_neighbor_triangle_original[point_trial[1]][placeholder_remake[1]][placeholder_remake_01[1]] = triangle_trial[0];
        point_neighbor_triangle_original[vertex_trial_opposite][placeholder_remake[2]][placeholder_remake_01[2]] = triangle_trial[1];
        point_neighbor_triangle_original[point_trial[0]][placeholder_remake[3]][placeholder_remake_01[3]] = triangle_trial[1];

        // Change point_triangle_list
        point_triangle_list_original[vertex_trial][placeholder_triangle1] = point_triangle_list_original[vertex_trial][point_triangle_max_original[vertex_trial]-1];
        point_triangle_list_original[vertex_trial][point_triangle_max_original[vertex_trial]-1] = -1;
        point_triangle_max_original[vertex_trial] -= 1;
        point_triangle_list_original[vertex_trial_opposite][placeholder_triangle2] = point_triangle_list_original[vertex_trial_opposite][point_triangle_max_original[vertex_trial_opposite]-1];
        point_triangle_list_original[vertex_trial_opposite][point_triangle_max_original[vertex_trial_opposite]-1] = -1;
        point_triangle_max_original[vertex_trial_opposite] -= 1;
        // Add points
        point_triangle_list_original[point_trial[0]][point_triangle_max_original[point_trial[0]]] = triangle_trial[1];
        point_triangle_list_original[point_trial[1]][point_triangle_max_original[point_trial[1]]] = triangle_trial[0];
        point_triangle_max_original[point_trial[0]] += 1;
        point_triangle_max_original[point_trial[1]] += 1;

        // Update energy values
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_opposite] = phi_vertex[vertex_trial_opposite];
        phi_vertex_original[point_trial[0]] = phi_vertex[point_trial[0]];
        phi_vertex_original[point_trial[1]] = phi_vertex[point_trial[1]];
        mean_curvature_vertex_original[vertex_trial] = mean_curvature_vertex[vertex_trial];
        mean_curvature_vertex_original[vertex_trial_opposite] = mean_curvature_vertex[vertex_trial_opposite];
        mean_curvature_vertex_original[point_trial[0]] = mean_curvature_vertex[point_trial[0]];
        mean_curvature_vertex_original[point_trial[1]] = mean_curvature_vertex[point_trial[1]];
        sigma_vertex_original[vertex_trial] = sigma_vertex[vertex_trial];
        sigma_vertex_original[vertex_trial_opposite] = sigma_vertex[vertex_trial_opposite];
        sigma_vertex_original[point_trial[0]] = sigma_vertex[point_trial[0]];
        sigma_vertex_original[point_trial[1]] = sigma_vertex[point_trial[1]];

        double Area_total_diff = 0.0;
        Area_total_diff += area_faces[triangle_trial[0]] - area_faces_original[triangle_trial[0]];
        Area_total_diff += area_faces[triangle_trial[1]] - area_faces_original[triangle_trial[1]];
        Area_diff_thread[thread_id][0] += Area_total_diff;

        area_faces_original[triangle_trial[0]] = area_faces[triangle_trial[0]];
        area_faces_original[triangle_trial[1]] = area_faces[triangle_trial[1]]; 
            
    }
    else {
        // cout << "Reject move at " << vertex_trial << endl;
        steps_rejected_tether_thread[thread_id][0] += 1;
        // Change new values to original values
        // Have all points needed, now just time to change triangle_list, point_neighbor_list, link_triangle_list, point_triangle_list, point_neighbor_max, and point_triangle_max
        // Change triangle_list
        triangle_list[triangle_trial[0]][0] = triangle_list_original[triangle_trial[0]][0];
        triangle_list[triangle_trial[0]][1] = triangle_list_original[triangle_trial[0]][1];
        triangle_list[triangle_trial[0]][2] = triangle_list_original[triangle_trial[0]][2];

        triangle_list[triangle_trial[1]][0] = triangle_list_original[triangle_trial[1]][0];
        triangle_list[triangle_trial[1]][1] = triangle_list_original[triangle_trial[1]][1];
        triangle_list[triangle_trial[1]][2] = triangle_list_original[triangle_trial[1]][2];

        // Change point_neighbor_list
        // Add points
        point_neighbor_list[vertex_trial][placeholder_neighbor1] = point_neighbor_list_original[vertex_trial][placeholder_neighbor1];
        point_neighbor_triangle[vertex_trial][placeholder_neighbor1][0] = point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][0];
        point_neighbor_triangle[vertex_trial][placeholder_neighbor1][1] = point_neighbor_triangle_original[vertex_trial][placeholder_neighbor1][1];        
        point_neighbor_list[vertex_trial][point_neighbor_max[vertex_trial]] = point_neighbor_list_original[vertex_trial][point_neighbor_max[vertex_trial]];
        point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]][0] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max[vertex_trial]][0];
        point_neighbor_triangle[vertex_trial][point_neighbor_max[vertex_trial]][1] = point_neighbor_triangle_original[vertex_trial][point_neighbor_max[vertex_trial]][1];
        point_neighbor_max[vertex_trial] += 1;
        point_neighbor_list[vertex_trial_opposite][placeholder_neighbor2] = point_neighbor_list_original[vertex_trial_opposite][placeholder_neighbor2];
        point_neighbor_triangle[vertex_trial_opposite][placeholder_neighbor2][0] = point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][0];
        point_neighbor_triangle[vertex_trial_opposite][placeholder_neighbor2][1] = point_neighbor_triangle_original[vertex_trial_opposite][placeholder_neighbor2][1];        
        point_neighbor_list[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]] = point_neighbor_list_original[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]];
        point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][0] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][0];
        point_neighbor_triangle[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][1] = point_neighbor_triangle_original[vertex_trial_opposite][point_neighbor_max[vertex_trial_opposite]][1];
        point_neighbor_max[vertex_trial_opposite] += 1;

        // Delete points
        point_neighbor_list[point_trial[0]][point_neighbor_max_original[point_trial[0]]] = -1;
        point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]-1][0] = -1;
        point_neighbor_triangle[point_trial[0]][point_neighbor_max[point_trial[0]]-1][1] = -1;
        point_neighbor_list[point_trial[1]][point_neighbor_max_original[point_trial[1]]] = -1;
        point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]-1][0] = -1;
        point_neighbor_triangle[point_trial[1]][point_neighbor_max[point_trial[1]]-1][1] = -1;
        point_neighbor_max[point_trial[0]] -= 1;
        point_neighbor_max[point_trial[1]] -= 1;

        // Points changed due to redefinitions in triangle_trial[0] and triangle_trial[1]
        point_neighbor_triangle[vertex_trial][placeholder_remake[0]][placeholder_remake_01[0]] = point_neighbor_triangle_original[vertex_trial][placeholder_remake[0]][placeholder_remake_01[0]];
        point_neighbor_triangle[point_trial[1]][placeholder_remake[1]][placeholder_remake_01[1]] = point_neighbor_triangle_original[point_trial[1]][placeholder_remake[1]][placeholder_remake_01[1]];
        point_neighbor_triangle[vertex_trial_opposite][placeholder_remake[2]][placeholder_remake_01[2]] = point_neighbor_triangle_original[vertex_trial_opposite][placeholder_remake[2]][placeholder_remake_01[2]];
        point_neighbor_triangle[point_trial[0]][placeholder_remake[3]][placeholder_remake_01[3]] = point_neighbor_triangle_original[point_trial[0]][placeholder_remake[3]][placeholder_remake_01[3]];

        // Change point_triangle_list
        point_triangle_list[vertex_trial][placeholder_triangle1] = point_triangle_list_original[vertex_trial][placeholder_triangle1];
        point_triangle_list[vertex_trial][point_triangle_max[vertex_trial]] = point_triangle_list_original[vertex_trial][point_triangle_max[vertex_trial]];
        point_triangle_max[vertex_trial] += 1;
        point_triangle_list[vertex_trial_opposite][placeholder_triangle2] = point_triangle_list_original[vertex_trial_opposite][placeholder_triangle2];
        point_triangle_list[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]] = point_triangle_list_original[vertex_trial_opposite][point_triangle_max[vertex_trial_opposite]];
        point_triangle_max[vertex_trial_opposite] += 1;
        // Delete points
        point_triangle_list[point_trial[0]][point_triangle_max_original[point_trial[0]]] = -1;
        point_triangle_list[point_trial[1]][point_triangle_max_original[point_trial[1]]] = -1;
        point_triangle_max[point_trial[0]] -= 1;
        point_triangle_max[point_trial[1]] -= 1;

        // Update energy values
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_opposite] = phi_vertex_original[vertex_trial_opposite];
        phi_vertex[point_trial[0]] = phi_vertex_original[point_trial[0]];
        phi_vertex[point_trial[1]] = phi_vertex_original[point_trial[1]];
        mean_curvature_vertex[vertex_trial] = mean_curvature_vertex_original[vertex_trial];
        mean_curvature_vertex[vertex_trial_opposite] = mean_curvature_vertex_original[vertex_trial_opposite];
        mean_curvature_vertex[point_trial[0]] = mean_curvature_vertex_original[point_trial[0]];
        mean_curvature_vertex[point_trial[1]] = mean_curvature_vertex_original[point_trial[1]];
        sigma_vertex[vertex_trial] = sigma_vertex_original[vertex_trial];
        sigma_vertex[vertex_trial_opposite] = sigma_vertex_original[vertex_trial_opposite];
        sigma_vertex[point_trial[0]] = sigma_vertex_original[point_trial[0]];
        sigma_vertex[point_trial[1]] = sigma_vertex_original[point_trial[1]];
   
        area_faces[triangle_trial[0]] = area_faces_original[triangle_trial[0]];
        area_faces[triangle_trial[1]] = area_faces_original[triangle_trial[1]]; 

        
    }
    steps_tested_tether_thread[thread_id][0] += 1;
    // cout << "End tether" << endl;

	/*
    Phi_ = Phi;
	initializeEnergy();
	cout.precision(17);
	cout << "Phi after chance in tether cut" << std::scientific << (Phi - Phi_) << endl;
	*/
    /* 
    cout << "At end of tether cut!" << endl;
    int loop[4] = {vertex_trial, vertex_trial_opposite, point_trial[0], point_trial[1]};
    cout << "Triangle 1: " << triangle_list_original[triangle_trial[0]][0] << " " << triangle_list_original[triangle_trial[0]][1] << " " << triangle_list_original[triangle_trial[0]][2] << endl;
    cout << "Triangle 2: " << triangle_list_original[triangle_trial[1]][0] << " " << triangle_list_original[triangle_trial[1]][1] << " " << triangle_list_original[triangle_trial[1]][2] << endl;
    cout << "Triangle 1: " << triangle_list[triangle_trial[0]][0] << " " << triangle_list[triangle_trial[0]][1] << " " << triangle_list[triangle_trial[0]][2] << endl;
    cout << "Triangle 2: " << triangle_list[triangle_trial[1]][0] << " " << triangle_list[triangle_trial[1]][1] << " " << triangle_list[triangle_trial[1]][2] << endl;
    for(int j_loop=0; j_loop<4; j_loop++) {
        int i = loop[j_loop];
        cout << "Max neighbors at " << i << " is " << point_neighbor_max[i] << endl;
        cout << "Max triangles at " << i << " is " << point_triangle_max[i] << endl;
    }

    
    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j_loop=0; j_loop<4; j_loop++) { 
            int i = loop[i_loop];
            int j = loop[j_loop];
            cout << "Link list at " << i << " " << j << " is 1: " << link_triangle_list[i][j][0] << " and 2: " << link_triangle_list[i][j][1] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j=0; j<10; j++){
            int i = loop[i_loop];
            cout << "Neighbor list at " << i << " entry " << j << " :" << point_neighbor_list[i][j] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        int i = loop[i_loop];
        cout << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<10; j++) {
            cout << point_triangle_list[i][j] << " ";
        }
        cout << endl;
    }
    
    for(int j_loop=0; j_loop<4; j_loop++) {
        int i = loop[j_loop];
        cout << "Max neighbors at " << i << " is " << point_neighbor_max_original[i] << endl;
        cout << "Max triangles at " << i << " is " << point_triangle_max_original[i] << endl;
    }

    
    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j_loop=0; j_loop<4; j_loop++) { 
            int i = loop[i_loop];
            int j = loop[j_loop];
            cout << "Link list at " << i << " " << j << " is 1: " << link_triangle_list_original[i][j][0] << " and 2: " << link_triangle_list_original[i][j][1] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        for(int j=0; j<10; j++){
            int i = loop[i_loop];
            cout << "Neighbor list at " << i << " entry " << j << " :" << point_neighbor_list_original[i][j] << endl;
        }
    }

    for(int i_loop=0; i_loop<4; i_loop++) {
        int i = loop[i_loop];
        cout << "Triangle list at vertex " << i << " is given by ";
        for(int j=0; j<10; j++) {
            cout << point_triangle_list_original[i][j] << " ";
        }
        cout << endl;
    }
    */
    
}

void ChangeMassNonCon(int vertex_trial, int thread_id) {
    // Pick random site and change the Ising array value
    // Non-mass conserving
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
    // Have to have implementation that chooses from within the same checkerboard set
    if(Ising_Array[vertex_trial] == 2) {
        // If protein node is selected, do a moveProtein_gen move instead
        moveProtein_gen(vertex_trial, thread_id);
        return;
	} 
    // Change spin
    int Ising_Array_trial = 0;
    if(Ising_Array[vertex_trial] == 0){
        Ising_Array_trial = 1;
    }

    double Phi_diff_phi = 0;

    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    double Phi_diff_mag = -h_external*(ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial]*sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    double Phi_diff = Phi_diff_phi+Phi_diff_bending+Phi_diff_mag;
    double Chance = local_generator.d();
    if(Chance<exp(-Phi_diff/T)){
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Mass_diff_thread[thread_id][0] += (Ising_Array_trial-Ising_Array[vertex_trial]);
        Magnet_diff_thread[thread_id][0] += (ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);
        Ising_Array[vertex_trial] = Ising_Array_trial;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;
}

void ChangeMassNonCon_gl(int vertex_trial) {
    // Pick random site and change the Ising array value per Glauber dynamics
    // Non-mass conserving
    Saru& local_generator = generators[omp_get_thread_num()];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
    while (Ising_Array[vertex_trial] == 2) {
		vertex_trial = local_generator.rand_select(vertices-1);
	}
    // Change spin
    int Ising_Array_trial = 0;
    if(Ising_Array[vertex_trial] == 0){
        Ising_Array_trial = 1;
    }

    double Phi_magnet = 0;

    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial];
        Phi_magnet -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // External field effect
    Phi_magnet -= h_external*(ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial]*sigma_vertex[vertex_trial]*diff_curv*diff_curv;
    double Phi_diff = Phi_magnet + phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    double Chance = local_generator.d();
    if(Chance<(1.0/(1.0+exp(Phi_diff/T)))) {
        #pragma omp atomic
        Phi += Phi_diff;
        #pragma omp atomic
        Mass += (Ising_Array_trial-Ising_Array[vertex_trial]);
        #pragma omp atomic
        Magnet += (ising_values[Ising_Array_trial]-ising_values[Ising_Array[vertex_trial]]);
        Ising_Array[vertex_trial] = Ising_Array_trial;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
    }
    
    else {
        #pragma omp atomic
        steps_rejected_mass += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];    
    }
    #pragma omp atomic
    steps_tested_mass += 1;
}

void ChangeMassCon(int vertex_trial, int thread_id) {
    // Pick random site and attemp to swap Ising array value with array in random nearest neighbor direction
    // Mass conserving
    Saru& local_generator = generators[thread_id];
    if(vertex_trial == -1) {
        vertex_trial = local_generator.rand_select(vertices-1);
    }
    if(Ising_Array[vertex_trial] == 2) {
        // If protein node is selected, do a moveProtein_gen move instead
        moveProtein_gen(vertex_trial, thread_id);
        return;
	} 
    // Pick random direction
    int link_trial = local_generator.rand_select(point_neighbor_max[vertex_trial]-1);
    int vertex_trial_opposite = point_neighbor_list[vertex_trial][link_trial];

    // For now reject if the neighboring sites have the same array value or protein type
    if(Ising_Array[vertex_trial] == Ising_Array[vertex_trial_opposite]) {
        steps_rejected_mass_thread[thread_id][0] += 1;
        steps_tested_mass_thread[thread_id][0] += 1;
        return;
    }

    // Check to see if vertex_trial_opposite is not in same checkerboard set
    if(checkerboard_index[vertex_trial] != checkerboard_index[vertex_trial_opposite]) {
        steps_rejected_mass_thread[thread_id][0] += 1;
    	steps_tested_mass_thread[thread_id][0] += 1;
		return;
    }

    // Set trial values
    int Ising_Array_trial_1 = Ising_Array[vertex_trial_opposite];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_diff_phi = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_opposite]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial_opposite]][Ising_Array[point_neighbor_list[vertex_trial_opposite][j]]]*ising_values[Ising_Array[vertex_trial_opposite]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[vertex_trial_opposite][j]]]*ising_values[Ising_Array_trial_2];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial_opposite][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[vertex_trial_opposite]-spon_curv[Ising_Array_trial_2];
    phi_vertex[vertex_trial_opposite] = k_b[Ising_Array_trial_2]*sigma_vertex[vertex_trial_opposite]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    Phi_diff_phi += (J_coupling[Ising_Array_trial_1][Ising_Array[vertex_trial_opposite]]*ising_values[Ising_Array_trial_1]-J_coupling[Ising_Array[vertex_trial]][Ising_Array[vertex_trial_opposite]]*ising_values[Ising_Array[vertex_trial]])*ising_values[Ising_Array[vertex_trial_opposite]];
    Phi_diff_phi += (J_coupling[Ising_Array_trial_2][Ising_Array[vertex_trial]]*ising_values[Ising_Array_trial_2]-J_coupling[Ising_Array[vertex_trial_opposite]][Ising_Array[vertex_trial]]*ising_values[Ising_Array[vertex_trial_opposite]])*ising_values[Ising_Array[vertex_trial]];

    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff_bending += phi_vertex[vertex_trial_opposite] - phi_vertex_original[vertex_trial_opposite];
    double Phi_diff = Phi_diff_bending+Phi_diff_phi;
    double Chance = local_generator.d();
    if(Chance<exp(-Phi_diff/T)){
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[vertex_trial_opposite] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_opposite] = phi_vertex[vertex_trial_opposite];
    }
    
    else {
        steps_rejected_mass_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_opposite] = phi_vertex_original[vertex_trial_opposite];    
    }
    steps_tested_mass_thread[thread_id][0] += 1;

}

void ChangeMassCon_nl() {
    // Pick random site and attemp to swap Ising array value with another nonlocal site
    // Mass conserving
    // Can't think of a good parallel implementation for now
    Saru& local_generator = generators[omp_get_thread_num()];
    int vertex_trial = local_generator.rand_select(vertices-1);
    // Pick random site with opposite spin
    int vertex_trial_2 = local_generator.rand_select(vertices-1);

    // Keep generating new trial values if Ising values are the same
    while((Ising_Array[vertex_trial] == Ising_Array[vertex_trial_2]) || (Ising_Array[vertex_trial] == 2) || (Ising_Array[vertex_trial_2] == 2)) {
    	vertex_trial = local_generator.rand_select(vertices-1);
        vertex_trial_2 = local_generator.rand_select(vertices-1);   		 
    }    

    // Set trial values
    int Ising_Array_trial_1 = Ising_Array[vertex_trial_2];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_magnet = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_magnet -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_2]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial_2]][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array[vertex_trial_2]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array_trial_2];
        Phi_magnet -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[vertex_trial_2]-spon_curv[Ising_Array_trial_2];
    phi_vertex[vertex_trial_2] = k_b[Ising_Array_trial_2]*sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    // Now check for self contribution
    // First check to see if sites are neighboring
    int check_double_count = link_triangle_test(vertex_trial, vertex_trial_2);
    if(check_double_count == 1) {
        Phi_magnet += (J_coupling[Ising_Array_trial_1][Ising_Array[vertex_trial_2]]*ising_values[Ising_Array_trial_1]-J_coupling[Ising_Array[vertex_trial]][Ising_Array[vertex_trial_2]]*ising_values[Ising_Array[vertex_trial]])*ising_values[Ising_Array[vertex_trial_2]];
        Phi_magnet += (J_coupling[Ising_Array_trial_2][Ising_Array[vertex_trial]]*ising_values[Ising_Array_trial_2]-J_coupling[Ising_Array[vertex_trial_2]][Ising_Array[vertex_trial]]*ising_values[Ising_Array[vertex_trial_2]])*ising_values[Ising_Array[vertex_trial]];
    }    

    double Phi_diff = Phi_magnet + phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff += phi_vertex[vertex_trial_2] - phi_vertex_original[vertex_trial_2];
    double Chance = local_generator.d();
    if(Chance<exp(-Phi_diff/T)){
        Phi += Phi_diff;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[vertex_trial_2] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_2] = phi_vertex[vertex_trial_2];
    }
    
    else {
        steps_rejected_mass += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_2] = phi_vertex_original[vertex_trial_2];    
    }
    steps_tested_mass += 1;

}

void moveProtein_gen(int vertex_trial, int thread_id) {
    // Pick random protein and attempt to move it in the y-direction
    // As protein's not merging, don't let them
    Saru& local_generator = generators[thread_id];
    // cout << "Protein trial " << protein_trial << " ";
    // Pick direction
    // Instead just go with one it's neighbors
    int direction_trial = local_generator.rand_select(point_neighbor_max[vertex_trial]-1);
    int center_trial = point_neighbor_list[vertex_trial][direction_trial]; 
    // Reject if about to swap with another protein of the same type
    if((protein_node[vertex_trial] != -1) && (protein_node[vertex_trial] == protein_node[center_trial])) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Reject if not in the same checkerboard set
    if(checkerboard_index[vertex_trial] != checkerboard_index[center_trial]) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Have to break down into cases
    int case_vertex = protein_node[vertex_trial];
    int case_center = protein_node[center_trial];

    // Energetics of swapping those two
    // Set trial values
    int Ising_Array_trial_1 = Ising_Array[center_trial];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_diff_phi = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[center_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[center_trial]][Ising_Array[point_neighbor_list[center_trial][j]]]*ising_values[Ising_Array[center_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[center_trial][j]]]*ising_values[Ising_Array_trial_2];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[center_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[center_trial]-spon_curv[Ising_Array_trial_2];
    phi_vertex[center_trial] = k_b[Ising_Array_trial_2]*sigma_vertex[center_trial]*diff_curv_c*diff_curv_c;

    // Now add in self contribution to cancel that out 
    Phi_diff_phi += (J_coupling[Ising_Array_trial_1][Ising_Array[center_trial]]*ising_values[Ising_Array_trial_1]-J_coupling[Ising_Array[vertex_trial]][Ising_Array[center_trial]]*ising_values[Ising_Array[vertex_trial]])*ising_values[Ising_Array[center_trial]];
    Phi_diff_phi += (J_coupling[Ising_Array_trial_2][Ising_Array[vertex_trial]]*ising_values[Ising_Array_trial_2]-J_coupling[Ising_Array[center_trial]][Ising_Array[vertex_trial]]*ising_values[Ising_Array[center_trial]])*ising_values[Ising_Array[vertex_trial]];
    
    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff_bending += phi_vertex[center_trial] - phi_vertex_original[center_trial];
    double Chance = local_generator.d();
    double db_factor = double(point_neighbor_max[vertex_trial])/double(point_neighbor_max[center_trial]);
    double Chance_factor = -T*log(Chance/db_factor);
    double Phi_diff = Phi_diff_phi+Phi_diff_bending;

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(Chance_factor>Phi_diff) {
        accept = true;
    }

    // cout << Phi_magnet << endl;
    // cout << Chance << " " << exp(-Phi_magnet/T) << endl;
    if(accept == true) {
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[center_trial] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[center_trial] = phi_vertex[center_trial];
        protein_node[vertex_trial] = case_center;
        protein_node[center_trial] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[center_trial] = phi_vertex_original[center_trial];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void moveProtein_nl(int vertex_trial, int vertex_trial_2, int thread_id) {
    // Pick two random nodes, see if one is a protein and attempt to swap if so
    Saru& local_generator = generators[thread_id];
    // cout << "Protein trial " << protein_trial << " ";
    // Reject if about to swap with another protein of the same type
    if(((protein_node[vertex_trial] == -1) || (protein_node[vertex_trial_2] == -1)) && (protein_node[vertex_trial] == protein_node[vertex_trial_2])) {
        steps_tested_protein_thread[thread_id][0]++;
        steps_rejected_protein_thread[thread_id][0]++;
        return;
    }
    // Have to break down into cases
    int case_vertex = protein_node[vertex_trial];
    int case_center = protein_node[vertex_trial_2];

    // Energetics of swapping those two
    // Set trial values
    int Ising_Array_trial_1 = Ising_Array[vertex_trial_2];
    int Ising_Array_trial_2 = Ising_Array[vertex_trial];

    // Now actually swap. We'll evaluate the energy difference using the neighboring stencil for each. Note this double counts trial, trial_2 so we'll need to add that part back in twice
    double Phi_diff_phi = 0;
    for(int j=0; j<point_neighbor_max[vertex_trial]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial]][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array[vertex_trial]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_1][Ising_Array[point_neighbor_list[vertex_trial][j]]]*ising_values[Ising_Array_trial_1];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_v = mean_curvature_vertex[vertex_trial]-spon_curv[Ising_Array_trial_1];
    phi_vertex[vertex_trial] = k_b[Ising_Array_trial_1]*sigma_vertex[vertex_trial]*diff_curv_v*diff_curv_v;

    // Trial 2
    for(int j=0; j<point_neighbor_max[vertex_trial_2]; j++) {
        double Site_diff = J_coupling[Ising_Array[vertex_trial_2]][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array[vertex_trial_2]];
        double Site_diff_2 = J_coupling[Ising_Array_trial_2][Ising_Array[point_neighbor_list[vertex_trial_2][j]]]*ising_values[Ising_Array_trial_2];
        Phi_diff_phi -= (Site_diff_2-Site_diff)*ising_values[Ising_Array[point_neighbor_list[vertex_trial_2][j]]];
    }    
    
    // Evaluate energy difference due to types swapping from mean curvature, surface tension
    double diff_curv_c = mean_curvature_vertex[vertex_trial_2]-spon_curv[Ising_Array_trial_2];
    phi_vertex[vertex_trial_2] = k_b[Ising_Array_trial_2]*sigma_vertex[vertex_trial_2]*diff_curv_c*diff_curv_c;

    double Phi_diff_bending = phi_vertex[vertex_trial] - phi_vertex_original[vertex_trial];
    Phi_diff_bending += phi_vertex[vertex_trial_2] - phi_vertex_original[vertex_trial_2];
    double Chance = local_generator.d();
    double Chance_factor = -T*log(Chance);
    double Phi_diff = Phi_diff_phi+Phi_diff_bending;

    // As switching curvatures, have to look at that closes
    bool accept = false;
    if(Chance_factor>Phi_diff) {
        accept = true;
    }

    // cout << Phi_magnet << endl;
    // cout << Chance << " " << exp(-Phi_magnet/T) << endl;
    if(accept == true) {
        Phi_diff_thread[thread_id][0] += Phi_diff;
        Phi_bending_diff_thread[thread_id][0] += Phi_diff_bending;
        Phi_phi_diff_thread[thread_id][0] += Phi_diff_phi;
        Ising_Array[vertex_trial] = Ising_Array_trial_1;
        Ising_Array[vertex_trial_2] = Ising_Array_trial_2;
        phi_vertex_original[vertex_trial] = phi_vertex[vertex_trial];
        phi_vertex_original[vertex_trial_2] = phi_vertex[vertex_trial_2];
        protein_node[vertex_trial] = case_center;
        protein_node[vertex_trial_2] = case_vertex;
    }
    else {
        steps_rejected_protein_thread[thread_id][0] += 1;
        phi_vertex[vertex_trial] = phi_vertex_original[vertex_trial];
        phi_vertex[vertex_trial_2] = phi_vertex_original[vertex_trial_2];
    }
    steps_tested_protein_thread[thread_id][0] += 1;
}

void ChangeArea() {
// Attempt to change Length_x and Length_y uniformly
// Going with logarthmic changes in area for now
// Naw, let's try discrete steps
// But discrete changes are also legitimate
    // Select trial scale_xy factor
    // double log_scale_xy_trial = log(scale_xy)+lambda_scale*randNeg1to1(generator);
    // double scale_xy_trial = exp(log_scale_xy_trial);
    chrono::steady_clock::time_point t1_area;
    chrono::steady_clock::time_point t2_area;
    t1_area = chrono::steady_clock::now();

    double scale_xy_trial = scale_xy+lambda_scale*generator.d(-1.0,1.0);
    if(scale_xy_trial <= 0.0) {
        steps_rejected_area++;
        steps_tested_area++;
        return;
    }
    Length_x = Length_x_base*scale_xy_trial;
    Length_y = Length_y_base*scale_xy_trial;
    t2_area = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_area-t1_area;
    time_storage_area[0] += time_span.count();
    // Reform neighbor list
    t1_area = chrono::steady_clock::now();
    generateNeighborList();
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[1] += time_span.count();
    // Store original values
    t1_area = chrono::steady_clock::now();
    double Phi_ = Phi;
    double Phi_bending_ = Phi_bending;
    double Phi_phi_ = Phi_phi;
    double Area_total_ = Area_total;
    // Recompute energy
    // Note that this version doesn't override all variables
    initializeEnergy_scale();
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[2] += time_span.count();
    // Now accept/reject
    t1_area = chrono::steady_clock::now();
    double Chance = generator.d();
    double Phi_diff = Phi-Phi_-T*2*vertices*log(scale_xy_trial/scale_xy);
    if((Chance<exp(-Phi_diff/T)) && (Phi_diff < pow(10,10))) {
        scale_xy = scale_xy_trial;
        Length_x_old = Length_x;
        Length_y_old = Length_y;
        box_x = Length_x/double(nl_x); 
        box_y = Length_y/double(nl_y); 
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            phi_vertex_original[i] = phi_vertex[i];
        }
        #pragma omp parallel for
        for(int i=0; i<faces; i++) {
            area_faces_original[i] = area_faces[i];
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            mean_curvature_vertex_original[i] = mean_curvature_vertex[i]; 
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            sigma_vertex_original[i] = sigma_vertex[i]; 
        }
    }
    else {
        steps_rejected_area++;
        Phi = Phi_;
        Phi_bending = Phi_bending_;
        Phi_phi = Phi_phi_;
        Area_total = Area_total_;
        Length_x = Length_x_old;
        Length_y = Length_y_old;
        generateNeighborList();
        box_x = Length_x/double(nl_x); 
        box_y = Length_y/double(nl_y); 
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            phi_vertex[i] = phi_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<faces; i++) {
            area_faces[i] = area_faces_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            mean_curvature_vertex[i] = mean_curvature_vertex_original[i];
        }
        #pragma omp parallel for
        for(int i=0; i<vertices; i++) {
            sigma_vertex[i] = sigma_vertex_original[i];
        }
    }
    steps_tested_area++;
    t2_area = chrono::steady_clock::now();
    time_span = t2_area-t1_area;
    time_storage_area[3] += time_span.count();
}

void CheckerboardMCSweep(bool nl_move) {
// Implementation idea
// Working in two dimensions for decomposition
// Why two instead of three?
// Makes more sense as that's how the cell be 
// So for that, the space will be divided per
//  ||| 2 ||| ||| 3 |||
//  ||| 0 ||| ||| 1 |||
// Repeated for the whole system
// Constraints on checkboard x/y: must be divisble by 2
// Will have an ideal size >> 1, and then round to nearest even number from there
// For the use of the checkerboard itself, use the following
// ALGORITHM
// Shuffle order of checkerboard C that will be iterated through
// For cells c in C(i) do
//  shuffle particle ordering in c(j)
//  Select random particle in c(j), perform random move
//  Reject move if it goes out of cell
// end for
// Shift cell and rebuilt this list
    chrono::steady_clock::time_point t1_displace;
    chrono::steady_clock::time_point t2_displace;
    // Generate checkerboard
    t1_displace = chrono::steady_clock::now();
    generateCheckerboard(); 
    t2_displace = chrono::steady_clock::now();
    chrono::duration<double> time_span = t2_displace-t1_displace;
    time_storage_displace[0] += time_span.count();
    // Create order of C
    t1_displace = chrono::steady_clock::now();
    vector<int> set_order = {0, 1, 2, 3};
    shuffle_saru(generator, set_order);
    // Pick number of moves to do per sweep
    // Let's go with 3 times average number of particles per cell
    int move_count = (3*vertices)/(checkerboard_x*checkerboard_y);
    // Int array that has standard modifications depending on what set we are working on
    int cell_modify_x[4] = {0,1,0,1};
    int cell_modify_y[4] = {0,0,1,1};
    // Have diff arrays to soter results to avoid atomic operations
    // Set to 0 here
    #pragma omp parallel for
    for(int i=0; i<active_threads; i++) {
        Phi_diff_thread[i][0] = 0;
        Phi_bending_diff_thread[i][0] = 0;
        Phi_phi_diff_thread[i][0] = 0;
        Area_diff_thread[i][0] = 0;
        Mass_diff_thread[i][0] = 0;
        Magnet_diff_thread[i][0] = 0;
        steps_tested_displace_thread[i][0] = 0;
        steps_rejected_displace_thread[i][0] = 0;
        steps_tested_tether_thread[i][0] = 0;
        steps_rejected_tether_thread[i][0] = 0;
        steps_tested_mass_thread[i][0] = 0;
        steps_rejected_mass_thread[i][0] = 0;
        steps_tested_protein_thread[i][0] = 0;
        steps_rejected_protein_thread[i][0] = 0;
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    time_storage_displace[1] += time_span.count();
    // Now iterate through elements of checkerboard_set
    t1_displace = chrono::steady_clock::now();
    for(int i=0; i<4; i++) {
        // Loop through cells in set_order[i]
        // Note by construction checkerboard_x*checkerboard_y/4 elements
        #pragma omp parallel for
        for(int j=0; j<checkerboard_x*checkerboard_y/4; j++) { 
            Saru& local_generator = generators[omp_get_thread_num()];
            int thread_id = omp_get_thread_num();
            // Figure out which cell this one is working on
            // Have base of
            int cell_x = j%(checkerboard_x/2);
            int cell_y = j/(checkerboard_x/2);
            // Now use cell modify arrays to figure out what cell we on
            int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*checkerboard_x;
            // Now select random particles and perform random moves on them
            // Use new distribution functions to do the random selection
            // Determine nearest power to cell_current size
            unsigned int cell_size = checkerboard_list[cell_current].size();
            cell_size = cell_size - 1;
            cell_size = cell_size | (cell_size >> 1);
            cell_size = cell_size | (cell_size >> 2);
            cell_size = cell_size | (cell_size >> 4);
            cell_size = cell_size | (cell_size >> 8);
            cell_size = cell_size | (cell_size >> 16);
            for(int k=0; k<move_count; k++) {
                // Random particle
                int vertex_trial = local_generator.rand_select(checkerboard_list[cell_current].size()-1, cell_size);
                // cout << vertex_trial << endl;
                // cout << k << endl;
                // cout << cell_current << endl;
                // cout << checkerboard_list[cell_current].size() << endl;
                vertex_trial = checkerboard_list[cell_current][vertex_trial];
                // Random move
                double Chance = local_generator.d();
                if(Chance < 1.0/3.0) {
                    DisplaceStep(vertex_trial, thread_id);
                }
                else if ((Chance >= 1.0/3.0) && (Chance < 2.0/3.0)) {
                    TetherCut(vertex_trial, thread_id);
                }
                else {
                    ChangeMassNonCon(vertex_trial, thread_id);
                }
            }
            // dumpXYZCheckerboard("config_checker.xyz");
        }
    }
    // Now do nonlocal protein moves
    if (nl_move) {
        for(int i=0; i<4; i++) {
            // Construct pairs
            // Do so by shuffling a list of all cells
            vector<int> cells_possible;
            for(int j=0; j<checkerboard_x*checkerboard_y/4; j++) {
                // Figure out which cell this one is working on
                // Have base of
                int cell_x = j%(checkerboard_x/2);
                int cell_y = j/(checkerboard_x/2);
                // Now use cell modify arrays to figure out what cell we on
                int cell_current = (2*cell_x+cell_modify_x[set_order[i]])+(2*cell_y+cell_modify_y[set_order[i]])*checkerboard_x;
                cells_possible.push_back(cell_current);
            }
            shuffle_saru(generator, cells_possible);
            #pragma omp parallel for
            for(int j=0; j<cells_possible.size()/2; j++) { 
                Saru& local_generator = generators[omp_get_thread_num()];
                int thread_id = omp_get_thread_num();
                int cells_0 = cells_possible[2*j];
                int cells_1 = cells_possible[2*j+1];
                // Determine nearest power to cell_current size
                unsigned int cell_size_0 = checkerboard_list[cells_0].size();
                cell_size_0 = cell_size_0 - 1;
                cell_size_0 = cell_size_0 | (cell_size_0 >> 1);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 2);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 4);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 8);
                cell_size_0 = cell_size_0 | (cell_size_0 >> 16);
                unsigned int cell_size_1 = checkerboard_list[cells_1].size();
                cell_size_1 = cell_size_1 - 1;
                cell_size_1 = cell_size_1 | (cell_size_1 >> 1);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 2);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 4);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 8);
                cell_size_1 = cell_size_1 | (cell_size_1 >> 16);
                for(int k=0; k<move_count; k++) {
                    // Random particle
                    int vertex_trial = local_generator.rand_select(checkerboard_list[cells_0].size()-1, cell_size_0);
                    int vertex_trial_2 = local_generator.rand_select(checkerboard_list[cells_1].size()-1, cell_size_1);
                    // cout << vertex_trial << endl;
                    // cout << k << endl;
                    // cout << cell_current << endl;
                    // cout << checkerboard_list[cell_current].size() << endl;
                    vertex_trial = checkerboard_list[cells_0][vertex_trial];
                    vertex_trial_2 = checkerboard_list[cells_1][vertex_trial_2];
                    moveProtein_nl(vertex_trial, vertex_trial_2, thread_id);
                }
                // dumpXYZCheckerboard("config_checker.xyz");
            }
        }
    }
    #pragma omp parallel for reduction(+:Phi,Phi_bending,Phi_phi,Area_total,Mass,Magnet,steps_rejected_displace,steps_tested_displace,steps_rejected_tether,steps_tested_tether,steps_rejected_mass,steps_tested_mass,steps_rejected_protein,steps_tested_protein)
    for(int i=0; i<active_threads; i++) {
        Phi += Phi_diff_thread[i][0];
        Phi_bending += Phi_bending_diff_thread[i][0];
        Phi_phi += Phi_phi_diff_thread[i][0];
        Area_total += Area_diff_thread[i][0];
        Mass += Mass_diff_thread[i][0];
        Magnet += Magnet_diff_thread[i][0];
        steps_rejected_displace += steps_rejected_displace_thread[i][0];
        steps_tested_displace += steps_tested_displace_thread[i][0];
        steps_rejected_tether += steps_rejected_tether_thread[i][0];
        steps_tested_tether += steps_tested_tether_thread[i][0];
        steps_rejected_mass += steps_rejected_mass_thread[i][0];
        steps_tested_mass += steps_tested_mass_thread[i][0];
        steps_rejected_protein += steps_rejected_protein_thread[i][0];
        steps_tested_protein += steps_tested_protein_thread[i][0];
    }
    t2_displace = chrono::steady_clock::now();
    time_span = t2_displace-t1_displace;
    time_storage_displace[2] += time_span.count();
}

void nextStepSerial() {
// Pick step at random with given frequencies.
    // sigma_i_total = 0.0;
	/*
    double Chance = rand0to1(generator);
    if(Chance <= 0.5) {
        DisplaceStep();
    }
    else {
        TetherCut();        
    }
	*/
    double area_chance = 1.0/double(vertices);
    double chance_else = (1.0-area_chance)/3.0;
    for(int i=0; i<3; i++) {
        double Chance = generator.d();
        if(Chance < 1.0*chance_else) {
	        DisplaceStep();
        }
        else if ((Chance >= 1.0*chance_else) && (Chance < 2.0*chance_else)) {
            TetherCut();
        }
        else if ((Chance >= 2.0*chance_else) && (Chance < 1.0-area_chance)) {
            ChangeMassNonCon();
        }
        else {
            ChangeArea();
        }
    }
  // ChangeMassCon_nl();   
    // ChangeMassNonCon();
    // ChangeMassNonCon_gl();
    /*
	for(int i=0; i<10; i++) {
	    ChangeMassCon();
	}
    */
    // ChangeMassCon_nl();
    // cout << "Sigma_i_total is now " << sigma_i_total << endl;
}

void nextStepParallel(bool nl_move) {
// Pick step at random with given frequencies.
// Only two options for now is Checkerboard Sweep or Area change
    double area_chance = 0.5;
    double Chance = generator.d();
    if(Chance < (1-area_chance)) {
        t1 = chrono::steady_clock::now();
        CheckerboardMCSweep(nl_move);
        t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        time_storage_cycle[0] += time_span.count();
    }
    else {
        t1 = chrono::steady_clock::now();
        ChangeArea();
        t2 = chrono::steady_clock::now();
        chrono::duration<double> time_span = t2-t1;
        time_storage_cycle[1] += time_span.count();
    }
}

/*
void dumpConfig(string name) {
    ofstream myfile;
    myfile.open (name);
    for(int i=1; i <= dim_x; i++) {
        for(int j=1; j <= dim_y; j++) {
            myfile << i << " " << j << " " << IsingArray[i-1][j-1] << endl;
        }
    }
    myfile.close();
}
*/
void dumpXYZConfig(string name) {
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

void dumpXYZConfigNormal(string name) {
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

void dumpXYZCheckerboard(string name) {
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

void dumpPhiNode(string name){
    ofstream myfile;
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << "# Phi at each node at " << steps_tested_displace+steps_tested_tether+steps_tested_mass << endl;
    myfile << "# i Phi" << endl;
    for(int i=1; i <= vertices; i++){
        myfile << i << " " << std::scientific << phi_vertex[i-1] << endl;
    }
    myfile.close();
}

void dumpAreaNode(string name){
    ofstream myfile;
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << "# Area at each face at " << steps_tested_displace+steps_tested_tether+steps_tested_mass << endl;
    myfile << "# i Area" << endl;
    for(int i=1; i <= faces; i++){
        myfile << i << " " << std::scientific << area_faces[i-1] << endl;
    }
    myfile.close();
}

void sampleNumberNeighbors(int it) {
    for(int i=0; i<neighbor_max; i++) {
        numbers_neighbor[i][it] = 0;
    }
    for(int i=0; i<vertices; i++) {
        numbers_neighbor[point_neighbor_max[i]][it] += 1;
    }
}

void dumpNumberNeighbors(string name, int it) {
    ofstream myfile;
    myfile.open (output_path+"/"+name, std::ios_base::app);
    myfile << "# Neighbor at " << steps_tested_displace+steps_tested_tether+steps_tested_mass << endl;
    // Get normalization constant
    long long int number_total = 0;
    for(int i=0; i<neighbor_max; i++) {
        number_total += numbers_neighbor[i][it];
    }

    double numbers_norm[neighbor_max];
    for(int i=0; i<neighbor_max; i++) {
        numbers_norm[i] = double(numbers_neighbor[i][it])/double(number_total);
    }
   
    for(int i=0; i < neighbor_max; i++) {
        myfile << i << " " << std::scientific << numbers_norm[i] << endl;
    }
    myfile.close();
}

void equilibriate(int cycles, chrono::steady_clock::time_point& begin) {
// Simulate for number of steps
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    double spon_curv_step = 4*spon_curv_end/cycles;
    int i=0;
    while(time_span_m.count() < final_warning) {
        SaruSeed(count_step);
        count_step++;
        if(nl_move_start == 0) {
            nextStepParallel(false);
        }
        else {
            nextStepParallel(true);
        }
        if(i < cycles/4) {
            spon_curv[2] += spon_curv_step;
            initializeEnergy();
        }
        else if(i == cycles/4){
            spon_curv[2] = spon_curv_end;
        }
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
            double Phi_ = Phi;
            double Phi_bending_ = Phi_bending;
            double Phi_phi_ = Phi_phi;
            initializeEnergy();
            cout.rdbuf(myfilebuf);
			cout << "Cycle " << i << endl;
			cout << "Energy " << std::scientific << Phi << " " << std::scientific << Phi-Phi_ << endl;
            cout << "Phi_bending " << std::scientific << Phi_bending << " " << std::scientific << Phi_bending-Phi_bending_ << " Phi_phi " << std::scientific << Phi_phi << " " << std::scientific << Phi_phi-Phi_phi_ << endl;
            cout << "Mass " << Mass << endl;
            cout << "Area " << Area_total << " and " << Length_x*Length_y << endl;
            cout << "spon_curv " << spon_curv[2] << endl;
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[0] += time_span.count();

            t1_other = chrono::steady_clock::now();
			linkMaxMin();
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[1] += time_span.count();

            t1_other = chrono::steady_clock::now();
            cout << "Displace " << steps_rejected_displace << "/" << steps_tested_displace << endl;
            cout << "Tether " << steps_rejected_tether << "/" << steps_tested_tether << endl;
            cout << "Mass " << steps_rejected_mass << "/" << steps_tested_mass << endl;
            cout << "Protein " << steps_rejected_protein << "/" << steps_tested_protein << endl;
            cout << "Area " << steps_rejected_area << "/" << steps_tested_area << endl;
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[2] += time_span.count();
            // cout << "Percentage of rejected steps: " << steps_rejected_displace+steps_rejected_tether << "/" << steps_tested_displace+steps_tested_tether << endl;
			/*
			cout << "Max diff is " << max_diff << endl;
			cout << "Relative difference is " << relative_diff << endl;
			max_diff = -1;
			relative_diff = 0;
			*/
		}
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
			outputTriangulation("int.off");	
            if(i%40000==0) {
                outputTriangulation("int_2.off");	
            }
            if(i%4000==0) {
			    dumpXYZConfig("config_equil.xyz");
			    // dumpXYZConfigNormal("config_equil_normal.xyz");
			    // dumpXYZCheckerboard("config_equil_test.xyz");
			    outputTriangulationAppend("equil.off");	
            }
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[3] += time_span.count();
		}
        /*
        if(i%100000==0) {
            outputTriangulationStorage();
        }
		if(i%40000==0) {
			dumpPhiNode("phinode_equil.txt");
            dumpAreaNode("areanode_equil.txt");
		}
        */
        middle = chrono::steady_clock::now();
        time_span_m = middle-begin;
        i++;
        if(i >= cycles) {
            break;
        }
    }
    outputTriangulation("int.off");	
    steps_tested_eq = steps_tested_displace + steps_tested_tether + steps_tested_mass + steps_tested_protein + steps_tested_area;
    steps_rejected_eq = steps_rejected_displace + steps_tested_tether + steps_rejected_mass + steps_rejected_protein + steps_rejected_area;
}

void simulate(int cycles, chrono::steady_clock::time_point& begin) {
// Simulate for number of cycles
    chrono::steady_clock::time_point t1_other;
    chrono::steady_clock::time_point t2_other;
    chrono::steady_clock::time_point middle;
    steps_tested_displace = 0;
    steps_rejected_displace = 0;
    steps_tested_tether = 0;
    steps_rejected_tether = 0;
    steps_tested_mass = 0;
    steps_rejected_mass = 0;
    steps_tested_protein = 0;
    steps_rejected_protein = 0;
    steps_tested_area = 0;
    steps_rejected_area = 0;
    ofstream myfile_umb;
    myfile_umb.precision(17);
    myfile_umb.open(output_path+"/mbar_data.txt", std::ios_base::app);
    cout.precision(8);
    middle = chrono::steady_clock::now();
    chrono::duration<double> time_span_m = middle-begin;
    // cout << "Total time: " << time_span.count() << " s" << endl;
    int i = 0;
    // for(int i=0; i<cycles; i++) {
    while(time_span_m.count() < final_warning) {
        SaruSeed(count_step);
        count_step++;
	    nextStepParallel(true);
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
            double Phi_ = Phi;
            double Phi_bending_ = Phi_bending;
            double Phi_phi_ = Phi_phi;
            initializeEnergy();
            cout.rdbuf(myfilebuf);
			cout << "Cycle " << i << endl;
			cout << "Energy " << std::scientific << Phi << " " << std::scientific << Phi-Phi_ << endl;
            cout << "Phi_bending " << std::scientific << Phi_bending << " " << std::scientific << Phi_bending-Phi_bending_ << " Phi_phi " << std::scientific << Phi_phi << " " << std::scientific << Phi_phi-Phi_phi_ << endl;
            cout << "Mass " << Mass << endl;
            cout << "Area " << Area_total << " and " << Length_x*Length_y << endl;
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[0] += time_span.count();

            t1_other = chrono::steady_clock::now();
			linkMaxMin();
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[1] += time_span.count();

            t1_other = chrono::steady_clock::now();
            cout << "Displace " << steps_rejected_displace << "/" << steps_tested_displace << endl;
            cout << "Tether " << steps_rejected_tether << "/" << steps_tested_tether << endl;
            cout << "Mass " << steps_rejected_mass << "/" << steps_tested_mass << endl;
            cout << "Protein " << steps_rejected_protein << "/" << steps_tested_protein << endl;
            cout << "Area " << steps_rejected_area << "/" << steps_tested_area << endl;
            t2_other = chrono::steady_clock::now();
            time_span = t2_other-t1_other;
            time_storage_other[2] += time_span.count();
            // cout << "Percentage of rejected steps: " << steps_rejected_displace+steps_rejected_tether << "/" << steps_tested_displace+steps_tested_tether << endl;
			/*
			cout << "Max diff is " << max_diff << endl;
			cout << "Relative difference is " << relative_diff << endl;
			max_diff = -1;
			relative_diff = 0;
			*/
		}
		if(i%1000==0) {
            t1_other = chrono::steady_clock::now();
			outputTriangulation("int.off");	
            if(count_step%20000==0) {
                outputTriangulation("int_2.off");	
            }
            if(i%4000==0) {
			    dumpXYZConfig("config.xyz");
			    // dumpXYZConfigNormal("config_normal.xyz");
			    outputTriangulationAppend("prod.off");	
			    // dumpXYZCheckerboard("config_test.xyz");
            }
            t2_other = chrono::steady_clock::now();
            chrono::duration<double> time_span = t2_other-t1_other;
            time_storage_other[3] += time_span.count();
		}
        /*
        if(i%100000==0) {
            outputTriangulationStorage();
        }
		if(i%40000==0) {
			dumpPhiNode("phinode.txt");
            dumpAreaNode("areanode.txt");
		}
        */
        if(i%storage_neighbor==0) {
            // sampleNumberNeighbors(i/storage_neighbor);  
            // dumpNumberNeighbors("numbers_dump.txt", i/storage_neighbor);
            // neighbors_counts++;
        }
        if(i%storage_umb_time==0) {
            energy_storage_umb[i/storage_umb_time] = Phi;
            umbOutput(i/storage_umb_time, myfile_umb);
            umb_counts++;
        }
        if(i%storage_time==0) {
		    energy_storage[i/storage_time] = Phi;
            area_storage[i/storage_time] = Area_total;
            area_proj_storage[i/storage_time] = Length_x*Length_y;
            mass_storage[i/storage_time] = Mass;
            storage_counts++;
        }
        middle = chrono::steady_clock::now();
        time_span_m = middle-begin;
        i++;
        if(i >= cycles) {
            break;
        }
        // dumpXYZConfig("config.xyz");
    }
    outputTriangulation("int.off");	
    steps_tested_prod = steps_tested_displace + steps_tested_tether + steps_tested_mass + steps_tested_protein + steps_tested_area;
    steps_rejected_prod = steps_rejected_displace + steps_tested_tether + steps_rejected_mass + steps_rejected_protein + steps_rejected_area;
    myfile_umb.close();
}


void energyAnalyzer() {
    // Evaluate average
    double energy_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        energy_ave += energy_storage[i];
    }
    energy_ave = energy_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double energy_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        energy_std += pow(energy_ave-energy_storage[i],2);
    }
    energy_std = sqrt(energy_std/(storage_counts-1));

    ofstream myfile;
    myfile.open (output_path+"/energy.txt", std::ios_base::app);
    myfile << "Energy from simulation run" << endl;
    myfile << "Average " << std::scientific << energy_ave << " Standard deviation " << std::scientific << energy_std << endl;
    myfile.close();

    myfile.open (output_path+"/energy_storage.txt", std::ios_base::app);
    myfile << "Energy from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << energy_storage[i] << endl;
    }
    myfile.close();
}

void areaAnalyzer() {
    // Evaluate average
    double area_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_ave += area_storage[i];
    }
    area_ave = area_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double area_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_std += pow(area_ave-area_storage[i],2);
    }
    area_std = sqrt(area_std/(storage_counts-1));

    ofstream myfile;
    myfile.open (output_path+"/area.txt", std::ios_base::app);
    myfile << "Area from simulation run" << endl;
    myfile << "Average " << std::scientific << area_ave << " Standard deviation " << std::scientific << area_std << endl;
    myfile.close();

    myfile.open (output_path+"/area_storage.txt", std::ios_base::app);
    myfile << "Area from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << area_storage[i] << endl;
    }
    myfile.close();
}

void areaProjAnalyzer() {
    // Evaluate average
    double area_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_ave += area_proj_storage[i];
    }
    area_ave = area_ave/storage_counts;
    Area_proj_average = area_ave;
    // Evaluate standard deviation using Bessel's correction
    double area_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        area_std += pow(area_ave-area_proj_storage[i],2);
    }
    area_std = sqrt(area_std/(storage_counts-1));

    ofstream myfile;
    myfile.open (output_path+"/area_proj.txt", std::ios_base::app);
    myfile << "Area from simulation run" << endl;
    myfile << "Average " << std::scientific << area_ave << " Standard deviation " << std::scientific << area_std << endl;
    myfile.close();

    myfile.open (output_path+"/area_proj_storage.txt", std::ios_base::app);
    myfile << "Area from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << area_proj_storage[i] << endl;
    }
    myfile.close();
}

void massAnalyzer() {
    // Evaluate average
    double mass_ave = 0.0;
    for(int i=0; i<storage_counts; i++) {
        mass_ave += mass_storage[i];
    }
    mass_ave = mass_ave/storage_counts;

    // Evaluate standard deviation using Bessel's correction
    double mass_std = 0.0;
    for(int i=0; i<storage_counts; i++) {
        mass_std += pow(mass_ave-mass_storage[i],2);
    }
    mass_std = sqrt(mass_std/(storage_counts-1));

    ofstream myfile;
    myfile.open (output_path+"/mass.txt", std::ios_base::app);
    myfile << "Mass from simulation run" << endl;
    myfile << "Average " << std::scientific << mass_ave << " Standard deviation " << std::scientific << mass_std << endl;
    myfile.close();

    myfile.open (output_path+"/mass_storage.txt", std::ios_base::app);
    myfile << "Mass from run" << endl;
    for(int i=0; i<storage_counts; i+=10) {
        myfile << std::scientific << mass_storage[i] << endl;
    }
    myfile.close();
}

void numberNeighborsAnalyzer() {
    int iter_total = neighbor_counts;
    long long int number_total[neighbor_max];
    long long int number_iter[iter_total];
    double number_neighbors_norm[neighbor_max][iter_total];
    double number_neighbors_norm_ave[neighbor_max];

    cout << "In neighbors analyzer" << endl;
    for(int j=0; j<neighbor_max; j++) {
        number_total[j] = 0;
    }

    for(int i=0; i<iter_total-1; i++) {
        number_iter[i] = 0;
        for(int j=neighbor_min; j<neighbor_max; j++) {
            number_total[j] += numbers_neighbor[j][i];
            number_iter[i] += numbers_neighbor[j][i];
        }
    }

    long long int number_total_all = 0;
    for(int i=0; i<neighbor_max; i++) {
        number_total_all += number_total[i];
        cout << i << " " << number_total[i] << endl;
    }
    cout << "number total " << number_total_all << endl;
    cout << "Out of neighbors analyzer" << endl;

    for(int i=0; i<iter_total-1; i++) {
        for(int j=neighbor_min; j<neighbor_max; j++) {
            number_neighbors_norm[j][i] = double(numbers_neighbor[j][i])/double(number_iter[i]);
        }
    }

    for(int i=neighbor_min; i<neighbor_max; i++) {
        number_neighbors_norm_ave[i] = double(number_total[i])/double(number_total_all);
    }

    // Evaluate standard deviation using Bessel's correction
    double number_neighbors_std[neighbor_max];
    for(int i=neighbor_min; i<neighbor_max; i++) {
        number_neighbors_std[i] = 0;
    }

    for(int i=0; i<iter_total-1; i++) {
        for(int j=neighbor_min; j<neighbor_max; j++) {
            number_neighbors_std[j] += pow(number_neighbors_norm_ave[j]-number_neighbors_norm[j][i],2);
        }
    }

    for(int i=neighbor_min; i<neighbor_max; i++) {
        number_neighbors_std[i] = sqrt(number_neighbors_std[i]/(iter_total-1));
    }

    ofstream myfile;
    myfile.open (output_path+"/number_neighbors_ave.txt", std::ios_base::app);
    myfile << "Neighbors from simulation run" << endl;
    for(int i=0; i<neighbor_max; i++) {
        myfile << i << " average " << std::scientific << number_neighbors_norm_ave[i] << " standard deviation " << std::scientific << number_neighbors_std[i] << endl;
    }
    myfile.close();
}

void umbAnalyzer() {
    // Dump umbrella sampling files
    // Evaluate average of energy to scale results
    double energy_ave_umb = 0;
    for(int i=0; i<umb_counts; i++) {
        energy_ave_umb += energy_storage_umb[i];
    }
    energy_ave_umb = energy_ave_umb/umb_counts;
    ofstream myfile;
    myfile.precision(17);
    myfile.open(output_path+"/mbar_data.txt", std::ios_base::app);
    for(int i=0; i<umb_counts; i++) {
        myfile << std::scientific << energy_harmonic_umb[i] << " " << std::scientific << energy_storage_umb[i] << endl;
    }
    myfile.close();
}

void umbOutput(int i, ofstream& myfile) {
    // Output current umbrella variables
    // Evaluate average of energy to scale results
    // Commands to speed things up
    // turns off synchronization of C++ streams
    ios_base::sync_with_stdio(false);
    // Turns off flushing of out before in
    cin.tie(NULL);
    myfile << std::scientific << energy_harmonic_umb[i] << " " << std::scientific << energy_storage_umb[i] << " " << std::scientific << Phi_bending << " " << std::scientific << Phi_phi << " " << std::scientific << Length_x*Length_y << " " << std::scientific << Area_total << "\n";
}
