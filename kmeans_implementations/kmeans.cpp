#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "helpers.hpp"
#include "serial.hpp"
#include "shared_cpu.hpp"
#include "kMeanCPUDistribute.hpp"
using namespace std;




int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //read args for number of clusters
    Args args(argc, argv);

    int dist_cpu_flag = args.dist_cpu ? 1 : 0;
    MPI_Bcast(&dist_cpu_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (dist_cpu_flag) {
        kMeanDistribute(argc,argv,100,args.k,args.input_file);
        MPI_Finalize();
        return 0;
    } 
    //read dataset and create point structs
    vector<Point> points = readData(args.input_file);

    //initialize centroids to share between kmeans implementations
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < args.k; ++i) {
        centroids.push_back(points[rand() % points.size()]);
    }

    if (rank == 0) {

        if (!args.skip_serial) {
            //perform serial kmeans
            vector<Point> serial_points = points; //copy points for serial implementation
            vector<Point> serial_centroids = centroids; //copy centroids for serial implementation
            performSerialKMeans(serial_points, 100, args.k, serial_centroids, args.output_dir);
        } 
        if (args.shared_cpu) {
            //perform shared cpu kmeans
            vector<Point> shared_points = points; 
            vector<Point> shared_centroids = centroids; 
            performSharedCPUKMeans(shared_points, 100, args.k, shared_centroids, args.output_dir, args.num_threads);
        } 
        if (args.cuda_gpu) {
            //call cuda gpu implementation

        } 

    }

    if (args.dist_gpu) {
        //call distributed gpu implementation
    }

    MPI_Finalize();
    return 0;
}