#include <cuda_runtime.h>
#include <string>
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
using std::vector;



void performDistributedGPUKMeans(vector<Point>& points, int epochs, int k, vector<Point>& centroids,
const std::string& output_dir, int threadsPerBlock
) {
    auto start = std::chrono::high_resolution_clock::now();
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    std::cout << "Using " << numDevices << " GPUs\n";


    // Splitting the data set
    int n = points.size();
    int chunkSize = (n + numDevices - 1) / numDevices;

    std::vector<std::vector<float>> localX(numDevices);
    std::vector<std::vector<float>> localY(numDevices);
    std::vector<std::vector<float>> localZ(numDevices);
    std::vector<int> localSize(numDevices);
    std::vector<int> localStart(numDevices);

    for (int device = 0; device < numDevices; device++) {
        int start = device * chunkSize;
        int end = std::min(start + chunkSize, n);
        int localsize = end - start;

        localStart[device] = start;
        localSize[device] = localsize;

        if (localsize <= 0) continue;

        localX[device].resize(localsize);
        localY[device].resize(localsize);
        localZ[device].resize(localsize);
        // Use GPU memory
        for (int i = 0; i < localsize; i++) {
            localX[device][i] = points[start + i].x;
            localY[device][i] = points[start + i].y;
            localZ[device][i] = points[start + i].z;
        }
    }

    // GPU buffer
    vector<float*> gpuX(numDevices), gpuY(numDevices), gpuZ(numDevices);
    vector<int*> gpu_clusters(numDevices);
    vector<float*> gpu_centroidX(numDevices);
    vector<float*> gpu_centroidY(numDevices);
    vector<float*> gpu_centroidZ(numDevices);
    vector<float*> gpu_sumX(numDevices);
    vector<float*> gpu_sumY(numDevices);
    vector<float*> gpu_sumZ(numDevices);
    vector<int*> gpu_counts(numDevices);

    // Host buffer centroid
    vector<vector<float>> centroidX_dev(numDevices, vector<float>(k));
    vector<vector<float>> centroidY_dev(numDevices, vector<float>(k));
    vector<vector<float>> centroidZ_dev(numDevices, vector<float>(k));

    // Allocate GPU memory
    for (int device = 0; device < numDevices; device++) {
        cudaSetDevice(device);
        int localsize = localSize[device];
        if (localsize <= 0) continue;

        cudaMalloc(&gpuX[device], localsize * sizeof(float));
        cudaMalloc(&gpuY[device], localsize * sizeof(float));
        cudaMalloc(&gpuZ[device], localsize * sizeof(float));
        cudaMalloc(&gpu_clusters[device], localsize * sizeof(int));
        cudaMalloc(&gpu_sumX[device], k * sizeof(float));
        cudaMalloc(&gpu_sumY[device], k * sizeof(float));
        cudaMalloc(&gpu_sumZ[device], k * sizeof(float));
        cudaMalloc(&gpu_counts[device], k * sizeof(int));
        cudaMalloc(&gpu_centroidX[device], k * sizeof(float));
        cudaMalloc(&gpu_centroidY[device], k * sizeof(float));
        cudaMalloc(&gpu_centroidZ[device], k * sizeof(float));

        // Copy point data
        cudaMemcpy(gpuX[device], localX[device].data(), localsize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuY[device], localY[device].data(), localsize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpuZ[device], localZ[device].data(), localsize * sizeof(float), cudaMemcpyHostToDevice);
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Make centroid array
        vector<float> centroidX(k);
        vector<float> centroidY(k);
        vector<float> centroidZ(k);

        for (int i = 0; i < k; i++) {
            centroidX[i] = centroids[i].x;
            centroidY[i] = centroids[i].y;
            centroidZ[i] = centroids[i].z;
        }

        // Send centroids to GPUs
        for (int device = 0; device < numDevices; device++) {
            cudaSetDevice(device);
            cudaMemcpy(gpu_centroidX[device], centroidX.data(), k * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_centroidY[device], centroidY.data(), k * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_centroidZ[device], centroidZ.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        }

        vector<std::thread> workers;

        // Capture local sums and counts inside the lambda
        vector<vector<float>> localSumX(numDevices, vector<float>(k, 0));
        vector<vector<float>> localSumY(numDevices, vector<float>(k, 0));
        vector<vector<float>> localSumZ(numDevices, vector<float>(k, 0));
        vector<vector<int>> localCounts(numDevices, vector<int>(k, 0));

        for (int device = 0; device < numDevices; device++) {
            workers.emplace_back([device, &gpuX, &gpuY, &gpuZ, &gpu_clusters, &gpu_centroidX,
                                  &gpu_centroidY, &gpu_centroidZ, &gpu_sumX, &gpu_sumY, &gpu_sumZ,
                                  &gpu_counts, &localSumX, &localSumY, &localSumZ, &localCounts,
                                  &localStart, &localSize, &centroids, k, threadsPerBlock
                                 ]() {
                cudaSetDevice(device);

                int localsize = localSize[device];
                if (localsize <= 0) return;

                int threads = threadsPerBlock;
                int numBlocks = (localsize + threads - 1) / threads;  // Fix 'size' to 'localsize'
                
                

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

                // Call kernel
                // Assign clusters
                cudaAssignClusters(d_x, d_y, d_z, d_clusters, d_centroidX, d_centroidY, d_centroidZ, k, localsize, threadsPerBlock);

                // Reset arrays
                cudaResetArrays(d_sumX, d_sumY, d_sumZ, d_counts, k,threadsPerBlock);

                // Accumulate centroids
                cudaAccumCentroids(d_x, d_y, d_z, d_clusters, d_sumX, d_sumY, d_sumZ, d_counts, localsize, k, threadsPerBlock);
                cudaDeviceSynchronize();
            });
        }

        for (auto& t : workers) {
            t.join();
        }

        // Merge the results from different devices and update centroids
        for (int i = 0; i < k; i++) {
            float sx = 0, sy = 0, sz = 0;
            for (int d = 0; d < numDevices; d++) {
                sx += localSumX[d][i];
                sy += localSumY[d][i];
                sz += localSumZ[d][i];
            }

            centroids[i].x = sx / numDevices;
            centroids[i].y = sy / numDevices;
            centroids[i].z = sz / numDevices;
        }
    }

    // Free GPU memory
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

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[Distributed GPU] k=" << k
              << ", time=" << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    writeOutput(points, output_dir + "/distributed_gpu_output.csv");
}