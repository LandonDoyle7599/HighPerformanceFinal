#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include "helpers.hpp"
#include "distributed_gpu.hpp"
#include "shared_gpu.hpp"
std::mutex mtx;

void performDistributedGPUKMeans(vector<Point>& points, int epochs, int k, vector<Point>& centroids,
const string& output_dir, int threadsPerBlock
) {
    auto start = chrono::high_resolution_clock::now();
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    cout << "Using " << numDevices << " GPUs\n";


    //spliting the data set
    int n = points.size();
    int chunkSize = (n + numDevices - 1) / numDevices;

    vector<float> globalSumX(k, 0);
    vector<float> globalSumY(k, 0);
    vector<float> globalSumZ(k, 0);
    vector<int> globalCounts(k, 0);
    
    vector<float*> gpuX(numDevices), gpuY(numDevices), gpuZ(numDevices);
    vector<int*> gpu_clusters(numDevices);

    vector<float*> gpu_centroidX(numDevices), gpu_centroidY(numDevices), gpu_centroidZ(numDevices);
    vector<float*> gpu_sumX(numDevices), gpu_sumY(numDevices), gpu_sumZ(numDevices);
    vector<int*> gpu_counts(numDevices);


    for (int device = 0; device < numDevices; device++) {
        cudaSetDevice(device);
        //form space before kmean
        int start = device * chunkSize;
        int end = min(start + chunkSize, n);
        int localsize = end - start;

        if (localsize <= 0) continue;

        cudaMalloc(&gpuX[device], localsize*sizeof(float));
        cudaMalloc(&gpuY[device], localsize*sizeof(float));
        cudaMalloc(&gpuZ[device], localsize*sizeof(float));
        cudaMalloc(&gpu_clusters[device], localsize*sizeof(int));

        cudaMalloc(&gpu_sumX[device], k*sizeof(float));
        cudaMalloc(&gpu_sumY[device], k*sizeof(float));
        cudaMalloc(&gpu_sumZ[device], k*sizeof(float));

        cudaMalloc(&gpu_centroidX[device], k*sizeof(float));
        cudaMalloc(&gpu_centroidY[device], k*sizeof(float));
        cudaMalloc(&gpu_centroidZ[device], k*sizeof(float));

        cudaMalloc(&gpu_counts[device], k*sizeof(int));
    }
    
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        fill(globalSumX.begin(), globalSumX.end(), 0);
        fill(globalSumY.begin(), globalSumY.end(), 0);
        fill(globalSumZ.begin(), globalSumZ.end(), 0);
        fill(globalCounts.begin(), globalCounts.end(), 0);
        
        vector<thread> workers;
        
        for (int device = 0; device < numDevices; device++) {
            workers.emplace_back([&, device]() {
                cudaSetDevice(device);
                //allocate GPU memory for this chunk
                int start = device * chunkSize;
                int end = min(start + chunkSize, n);
                int localsize = end - start;

                if (localsize <= 0) return;
            
                //convert points for GPU
                vector<float> localX(localsize);
                vector<float> localY(localsize);
                vector<float> localZ(localsize);
                for (int i= 0; i < localsize; i++) {
                    localX[i] = points[start + i].x;
                    localY[i] = points[start + i].y;
                    localZ[i] = points[start + i].z;
                }
                //copy centroids
                vector<float> centroidX(k);
                vector<float> centroidY(k);
                vector<float> centroidZ(k);

                for (int i = 0; i < k; i++) {
                    centroidX[i] = centroids[i].x;
                    centroidY[i] = centroids[i].y;
                    centroidZ[i] = centroids[i].z;
                }
                //  use gpu memory
                float* d_x = gpuX[device];
                float* d_y = gpuY[device];
                float* d_z = gpuZ[device];
                int* d_clusters = gpu_clusters[device];

                float* d_centroidX = gpu_centroidX[device];
                float* d_centroidY = gpu_centroidY[device];
                float* d_centroidZ = gpu_centroidZ[device];

                float* d_sumX = gpu_sumX[device];
                float* d_sumY = gpu_sumY[device];
                float* d_sumZ = gpu_sumZ[device];
                int* d_counts = gpu_counts[device];

                //gpu memory copy
               
                cudaMemcpy(d_x, localX.data(), localsize*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_y, localY.data(), localsize*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_z, localZ.data(), localsize*sizeof(float), cudaMemcpyHostToDevice);

                cudaMemcpy(d_centroidX, centroidX.data(), k*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_centroidY, centroidY.data(), k*sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_centroidZ, centroidZ.data(), k*sizeof(float), cudaMemcpyHostToDevice);

                //call kerenal
                int threads = threadsPerBlock;
                int numBlocks = (localsize + threads - 1) / threads;
                int numBlocksK = (k + threads - 1) / threads;

                int sharedMemSize = (3* k * sizeof(float)) + (k * sizeof(int));

                


                
                assignClusters<<<numBlocks, threads>>>(
                    d_x, d_y, d_z,
                    d_clusters,
                    d_centroidX, d_centroidY, d_centroidZ,
                    k, localsize
                    );
                
                 resetArrays<<<numBlocksK, threads>>>(
                    d_sumX, d_sumY, d_sumZ, d_counts, k
                );

                accumCentroids<<<numBlocks, threads, sharedMemSize>>>(
                    d_x, d_y, d_z,
                    d_clusters,
                    d_sumX, d_sumY, d_sumZ,
                    d_counts,
                    localsize, k
                );
                

                //mem copy back
                vector<float> localSumX(k);
                vector<float> localSumY(k);
                vector<float> localSumZ(k);
                vector<int> localCounts(k);
                
                cudaDeviceSynchronize();
                
                cudaMemcpy(localSumX.data(), d_sumX, k*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(localSumY.data(), d_sumY, k*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(localSumZ.data(), d_sumZ, k*sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(localCounts.data(), d_counts, k*sizeof(int), cudaMemcpyDeviceToHost);


                //combine into global
                {
                    lock_guard<mutex> lock(mtx);
                    for (int i = 0; i < k; i++) {
                        globalSumX[i] += localSumX[i];
                        globalSumY[i] += localSumY[i];
                        globalSumZ[i] += localSumZ[i];
                        globalCounts[i] += localCounts[i];
                    }
                }
                //copy clusters
                vector<int> localClusters(localsize);
                cudaMemcpy(localClusters.data(), d_clusters, localsize*sizeof(int), cudaMemcpyDeviceToHost);

                for (int i = 0; i < localsize; i++) {
                    points[start + i].cluster = localClusters[i];
                }
            
            });
        }
        
        for(auto& t : workers) {
            t.join();
        }
        //update centroids on CPU
        for (int i = 0; i < k; i++) {
            if (globalCounts[i] > 0) {
                centroids[i].x = globalSumX[i] / globalCounts[i];
                centroids[i].y = globalSumY[i] / globalCounts[i];
                centroids[i].z = globalSumZ[i] / globalCounts[i];
            }
        }
    
    }
    
    //free gpu memory
    for (int device = 0; device < numDevices; device++) {
        cudaSetDevice(device);

        cudaFree(gpuX[device]);
        cudaFree(gpuY[device]);
        cudaFree(gpuZ[device]);
        cudaFree(gpu_clusters[device]);

        cudaFree(gpu_centroidX[device]);
        cudaFree(gpu_centroidY[device]);
        cudaFree(gpu_centroidZ[device]);

        cudaFree(gpu_sumX[device]);
        cudaFree(gpu_sumY[device]);
        cudaFree(gpu_sumZ[device]);

        cudaFree(gpu_counts[device]);
    }
    
    auto end = chrono::high_resolution_clock::now();
    
    cout << "[Distributed GPU] k=" << k
     << ", time="
     << chrono::duration<double>(end - start).count()
     << " sec\n";

    writeOutput(points, output_dir + "/distributed_gpu_output.csv");
}
