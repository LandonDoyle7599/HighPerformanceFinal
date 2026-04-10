#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "helpers.hpp"
#include "serial.hpp"
#include "shared_cpu.hpp"
#include "shared_gpu.hpp"
#include "distributed_gpu.hpp"
#include "kMeanCPUDistribute.hpp"
using namespace std;




int main(int argc, char *argv[])
{
    //read args for number of clusters
    Args args(argc, argv);

    bool use_mpi = args.dist_cpu;

    if (use_mpi) {
        MPI_Init(&argc, &argv);
    }

    int rank = 0;
    if (use_mpi) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    //read dataset and create point structs
    vector<Point> points = readData(args.input_file);

    //initialize centroids to share between kmeans implementations
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < args.k; ++i) {
        centroids.push_back(points[rand() % points.size()]);
    }

    if (!args.skip_serial && rank == 0) {
        //perform serial kmeans
        vector<Point> serial_points = points; //copy points for serial implementation
        vector<Point> serial_centroids = centroids; //copy centroids for serial implementation
        performSerialKMeans(serial_points, args.epochs, args.k, serial_centroids, args.output_dir);
    } 
    if (use_mpi) MPI_Barrier(MPI_COMM_WORLD);

    if (args.shared_cpu && rank == 0) {
        //perform shared cpu kmeans
        vector<Point> shared_points = points; 
        vector<Point> shared_centroids = centroids; 
        performSharedCPUKMeans(shared_points, args.epochs, args.k, shared_centroids, args.output_dir, args.num_threads);
    } 
    if (use_mpi) MPI_Barrier(MPI_COMM_WORLD);
    if (args.dist_cpu) {
        //call distributed cpu implementation WITH shared data
        vector<Point> dist_points = points;
        vector<Point> dist_centroids = centroids;
        kMeanDistributePerformance(argc, argv, args.epochs, args.k, args.input_file, args.output_dir, dist_centroids, dist_points);
    }
    if (use_mpi) MPI_Barrier(MPI_COMM_WORLD);
    if (args.cuda_gpu && rank == 0) {
        //call cuda gpu implementation
        vector<Point> gpu_points = points;
        vector<Point> gpu_centroids = centroids;
        performSharedGPUKMeans(gpu_points, args.epochs, args.k, gpu_centroids, args.output_dir, args.threadsPerBlockCuda);
    } 
    if (use_mpi) MPI_Barrier(MPI_COMM_WORLD);
    if (args.dist_gpu && rank==0) {
        //call distributed gpu implementation
        vector<Point> dist_gpu_points = points;
        vector<Point> dist_gpu_centroids = centroids;
        performDistributedGPUKMeans(dist_gpu_points, args.epochs, args.k, dist_gpu_centroids, args.output_dir, args.threadsPerBlockDist, rank);
    }

    if (use_mpi) {
        MPI_Finalize();
    }
    return 0;
}