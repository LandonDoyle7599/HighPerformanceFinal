#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include "helpers.hpp"
#include "shared_cpu.hpp"
#include <omp.h>
using namespace std;




void sharedCPUKMeansClustering(vector<Point>& points, int epochs, int k, vector<Point>& centroids) {
    //kmeans
    vector<int> nPoints;
    vector<double> sumX, sumY, sumZ;
    vector<int> nPointsGlobal(k, 0);
    vector<double> sumXGlobal(k, 0), sumYGlobal(k, 0), sumZGlobal(k, 0);
    #pragma omp parallel default(none) shared(points, centroids, nPointsGlobal, sumXGlobal, sumYGlobal, sumZGlobal, k, epochs) private(nPoints, sumX, sumY, sumZ)
    {
        
        for (int epoch = 0; epoch < epochs; ++epoch){
            //reset local vectors for new epoch
            nPoints.assign(k, 0);
            sumX.assign(k, 0);
            sumY.assign(k, 0);
            sumZ.assign(k, 0);

            #pragma omp single //reset global vectors for new epoch
            {
                fill(nPointsGlobal.begin(), nPointsGlobal.end(), 0);
                fill(sumXGlobal.begin(), sumXGlobal.end(), 0);
                fill(sumYGlobal.begin(), sumYGlobal.end(), 0);
                fill(sumZGlobal.begin(), sumZGlobal.end(), 0);
            }

            //assign points to closest centroid
            #pragma omp for
            for (int i = 0; i < points.size(); ++i) {
                Point& p = points[i];
                for (int j = 0; j < k; ++j) {
                    double dist = p.distance(centroids[j]);
                    if (dist < p.minDist) {
                        p.minDist = dist;
                        p.cluster = j;
                    }
                }
            }

            //collect info on centroids
            #pragma omp for
            for (int i = 0; i < points.size(); ++i) {
                Point& p = points[i];
                int cluster = p.cluster;
                nPoints[cluster]++;
                sumX[cluster] += p.x;
                sumY[cluster] += p.y;
                sumZ[cluster] += p.z;
                p.minDist = DBL_MAX; //reset minDist for next epoch
            }

            //sum local values into global vectors
            for (int i = 0; i < k; ++i) {
                #pragma omp atomic
                nPointsGlobal[i] += nPoints[i];
                #pragma omp atomic
                sumXGlobal[i] += sumX[i];
                #pragma omp atomic
                sumYGlobal[i] += sumY[i];
                #pragma omp atomic
                sumZGlobal[i] += sumZ[i];
            }
            
            #pragma omp barrier //ensure all threads have updated global vectors before updating centroids
            #pragma omp single //update centroids
            {
                for (int i = 0; i < k; ++i) {
                    if (nPointsGlobal[i] > 0) {
                        centroids[i].x = sumXGlobal[i] / nPointsGlobal[i];
                        centroids[i].y = sumYGlobal[i] / nPointsGlobal[i];
                        centroids[i].z = sumZGlobal[i] / nPointsGlobal[i];
                    }
                }
            }
            //implicit barrier from single
        }
    }
}

void performSharedCPUKMeans(vector<Point>& points, int epochs,  int k, vector<Point>& centroids, const string& output_dir) {
    int num_threads;
    cout << "Enter the number of threads to use for the shared CPU implementation: ";
    cin >> num_threads;    
    omp_set_num_threads(num_threads);
    auto start_time = chrono::high_resolution_clock::now();
    sharedCPUKMeansClustering(points, epochs, k, centroids);
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;
    cout << "SharedCPUKMeans clustering with " << k << " clusters and " << num_threads << " threads took " << elapsed.count() << " seconds." << endl;
    writeOutput(points, output_dir + "shared_cpu_output.csv");
}






