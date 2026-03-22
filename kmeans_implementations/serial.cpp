#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include "helpers.hpp"
#include "serial.hpp"
using namespace std;




void serialKMeansClustering(vector<Point>& points, int epochs, int k, vector<Point>& centroids) {

    //kmeans
    for (int epoch = 0; epoch < epochs; ++epoch){
        //assign points to closest centroid
        for (auto& p: points) {
            for (int i = 0; i < k; ++i) {
                double dist = p.distance(centroids[i]);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = i;
                }
            }
        }

        //collect info on centroids
        vector<int> nPoints(k, 0);
        vector<double> sumX(k, 0), sumY(k, 0), sumZ(k, 0);
        
        for (const auto& p: points) {
            int cluster = p.cluster;
            nPoints[cluster]++;
            sumX[cluster] += p.x;
            sumY[cluster] += p.y;
            sumZ[cluster] += p.z;
        }

        //update centroids
        for (int i = 0; i < k; ++i) {
            if (nPoints[i] > 0) {
                centroids[i].x = sumX[i] / nPoints[i];
                centroids[i].y = sumY[i] / nPoints[i];
                centroids[i].z = sumZ[i] / nPoints[i];
            }
        }

        //reset minDist for next epoch
        for (auto& p: points) {
            p.minDist = DBL_MAX;
        }
    }
}

void performSerialKMeans(vector<Point>& points, int epochs, int k, vector<Point>& centroids, const string& output_dir) {
    auto start_time = chrono::high_resolution_clock::now();
    serialKMeansClustering(points, epochs, k, centroids);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "SerialKMeans clustering with " << k << " clusters took " << elapsed.count() << " seconds." << endl;
    writeOutput(points, output_dir + "/serial_output.csv");
}






