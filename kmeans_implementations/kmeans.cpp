#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "helpers.hpp"
#include "serial.hpp"
#include "shared_cpu.hpp"
#include "shared_gpu.hpp"
#include "kMeanCPUDistribute.hpp"
using namespace std;




int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //read args for number of clusters
    Args args(argc, argv);

    //read dataset and create point structs
    vector<Point> points = readData(args.input_file);

    //initialize centroids to share between kmeans implementations
    vector<Point> centroids;
    srand(0);
    for (int i = 0; i < args.k; ++i) {
        centroids.push_back(points[rand() % points.size()]);
    }

    if (!args.skip_serial && rank == 0) {
        //perform serial kmeans
        vector<Point> serial_points = points; //copy points for serial implementation
        vector<Point> serial_centroids = centroids; //copy centroids for serial implementation
        performSerialKMeans(serial_points, 100, args.k, serial_centroids, args.output_dir);
    } 

    if (args.shared_cpu && rank == 0) {
        //perform shared cpu kmeans
        vector<Point> shared_points = points; 
        vector<Point> shared_centroids = centroids; 
        performSharedCPUKMeans(shared_points, 100, args.k, shared_centroids, args.output_dir, args.num_threads);
    } 

    if (args.dist_cpu) {
        //call distributed cpu implementation WITH shared data
        kMeanDistribute(points, centroids, 100, args.k, args.output_dir);
    }

    if (args.cuda_gpu && rank == 0) {
        //call cuda gpu implementation
        vector<Point> gpu_points = points;
        vector<Point> gpu_centroids = centroids;
        performSharedGPUKMeans(gpu_points, 100, args.k, gpu_centroids, args.output_dir);
    } 

    if (args.dist_gpu) {
        //call distributed gpu implementation
    }

    MPI_Finalize();
    return 0;
}
