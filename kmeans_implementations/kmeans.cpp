#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "helpers.hpp"
#include "serial.hpp"
#include "shared_cpu.hpp"
#include "kMeanCPUDistribute.hpp"
using namespace std;




int main(int argc, char const *argv[])
{
    //read args for number of clusters
    Args args(argc, argv);
    if (args.dist_cpu) {
        kMeanDistribute(argc,argv,100,arg.k,args.input_file);
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

    if (args.dist_gpu) {
        //call distributed gpu implementation
    }
    return 0;
}