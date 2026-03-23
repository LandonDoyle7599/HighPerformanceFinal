#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "helpers.hpp"
#include "serial.hpp"
#include "shared_cpu.hpp"
using namespace std;


int main(int argc, char const *argv[])
{
    //read args for number of clusters
    int k;
    string input_file, output_dir;

    bool shared_cpu = false, cuda_gpu = false, dist_cpu = false, dist_gpu = false, skip_serial = false;
    if (argc >= 4) {
        k = atoi(argv[1]);
        input_file = argv[2];
        output_dir = argv[3];
        for (int i = 4; i < argc; ++i) {
            //TODO: adjust later to add flag specific arguments i.e. num_threads
            string flag = argv[i];
            if (flag == "--shared_cpu") shared_cpu = true;
            else if (flag == "--cuda_gpu") cuda_gpu = true;
            else if (flag == "--dist_cpu") dist_cpu = true;
            else if (flag == "--dist_gpu") dist_gpu = true;
            else if (flag == "--skip_serial") skip_serial = true;
        }
    } else {
        cout << "Usage: " << argv[0] << " <number_of_clusters> <input_file> <output_dir>" << endl;
        cout << "Optional flags to adjust what implementations are used: --shared_cpu, --cuda_gpu, --dist_cpu, --dist_gpu, --skip_serial" << endl;
        cout << "Be sure to add a trailing slash to the output directory path." << endl;
        return 1;
    }
    
    //read dataset and create point structs
    vector<Point> points = readData(input_file);

    //initialize centroids to share between kmeans implementations
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points[rand() % points.size()]);
    }

    if (!skip_serial) {
        //perform serial kmeans
        vector<Point> serial_points = points; //copy points for serial implementation
        vector<Point> serial_centroids = centroids; //copy centroids for serial implementation
        performSerialKMeans(serial_points, 100, k, serial_centroids, output_dir);
    } 
    if (shared_cpu) {
        //perform shared cpu kmeans
        vector<Point> shared_points = points; 
        vector<Point> shared_centroids = centroids; 
        performSharedCPUKMeans(shared_points, 100, k, shared_centroids, output_dir);
    } 
    if (cuda_gpu) {
        //call cuda gpu implementation
    } 
    if (dist_cpu) {
        //call distributed cpu implementation
    } 
    if (dist_gpu) {
        //call distributed gpu implementation
    }
    return 0;
}