#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cfloat>
#include "helpers.hpp"
#include "shared_gpu.hpp"
#include <chrono>
using namespace std;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) <<std::endl; \
        exit(1); \
    } \
}while(0)

__global__ void assignClusters(float* point_x, float* point_y, float* point_z,
    int* cluster,
    float* centroid_x, float* centroid_y, float* centroid_z,
    int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float minDist = FLT_MAX;
    int bestCluster = -1;

    for (int j = 0; j < k; j++) {
        float dist_x = point_x[i] - centroid_x[j];
        float dist_y = point_y[i] - centroid_y[j];
        float dist_z = point_z[i] - centroid_z[j];
        float dist = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;
        if (dist < minDist) {
            minDist = dist;
            bestCluster = j;
        }
    }
    cluster[i] = bestCluster;
}
__global__ void resetArrays(float* sumX, float* sumY, float* sumZ,int* counts, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        sumX[i] = 0.0;
        sumY[i] = 0.0;
        sumZ[i] = 0.0;
        counts[i] = 0;
    }
}
__global__ void accumCentroids(float* x, float* y, float* z, int* clusters,
    float* sumX, float* sumY, float* sumZ, int* counts, int n, int k)
{
    extern __shared__ char shared[];
    
    float* s_sumX =(float*)shared;
    float* s_sumY = (float*)&s_sumX[k];
    float* s_sumZ = (float*)&s_sumY[k];
    int* s_counts = (int*)&s_sumZ[k];
    
    int tid = threadIdx.x;
    
    //memory sharing
    for(int i = tid; i < k; i+= blockDim.x){
        s_sumX[i] = 0.0;
        s_sumY[i] = 0.0;
        s_sumZ[i] = 0.0;
        s_counts[i] = 0;
    }
    __syncthreads();
    
    int globalIdx = blockIdx.x * blockDim.x + tid;
    
    if(globalIdx < n){
        int c = clusters[globalIdx];
        if (c >= 0 && c < k) {
            atomicAdd(&s_sumX[c], x[globalIdx]);
            atomicAdd(&s_sumY[c], y[globalIdx]);
            atomicAdd(&s_sumZ[c], z[globalIdx]);
            atomicAdd(&s_counts[c], 1);
        }
    }
    __syncthreads();
    
    //to global memory
    for(int j = tid; j < k; j += blockDim.x){
         atomicAdd(&sumX[j], s_sumX[j]);
        atomicAdd(&sumY[j], s_sumY[j]);
        atomicAdd(&sumZ[j], s_sumZ[j]);
        atomicAdd(&counts[j], s_counts[j]);
    }
}
__global__ void updateCentroids(float* centroid_x, float* centroid_y, float* centroid_z,
    float* sumX, float* sumY, float* sumZ, int* counts, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k && counts[i] > 0) {
        centroid_x[i] = sumX[i]/counts[i];
        centroid_y[i] = sumY[i]/counts[i];
        centroid_z[i] = sumZ[i]/counts[i];
    }
}
//cuda wrappers
void cudaAssignClusters(float* point_x, float* point_y, float* point_z,
                                   int* cluster,
                                   float* centroid_x, float* centroid_y, float* centroid_z,
                                   int k, int n,int threadsPerBlock)
{
    
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    assignClusters<<<blocks, threadsPerBlock>>>(point_x, point_y, point_z, cluster,
                                                centroid_x, centroid_y, centroid_z, k, n);
    cudaDeviceSynchronize();
}

// Wrapper 2: resetArrays
void cudaResetArrays(float* sumX, float* sumY, float* sumZ, int* counts, int k,int threadsPerBlock)
{
    
    int blocks = (k + threadsPerBlock - 1) / threadsPerBlock;

    resetArrays<<<blocks, threadsPerBlock>>>(sumX, sumY, sumZ, counts, k);
    cudaDeviceSynchronize();
}

// Wrapper 3: accumCentroids
void cudaAccumCentroids(float* x, float* y, float* z, int* clusters,
                                   float* sumX, float* sumY, float* sumZ, int* counts,
                                   int n, int k, int threadsPerBlock)
{
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = 3 * k * sizeof(float) + k * sizeof(int);

    accumCentroids<<<blocks, threadsPerBlock, sharedMemSize>>>(x, y, z, clusters,
                                                         sumX, sumY, sumZ, counts, n, k);
    cudaDeviceSynchronize();
}
//timer funct to press use all the functions.
void performSharedGPUKMeans(vector<Point>& points, int epochs, int k, vector<Point>& centroids,
                            const string& output_dir, int threadsInBlock) {
    
    auto start_time = chrono::high_resolution_clock::now();

    performGPUKmeans(points, k, epochs, centroids, threadsInBlock);

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "Shared GPU KMeans clustering with "
         << k << " clusters took "
         << elapsed.count() << " seconds." << endl;

    writeOutput(points, output_dir + "/shared_gpu_output.csv");
}
void performGPUKmeans(vector<Point>& points, int k, int epochs, vector<Point>& centroids, int threadsInBlock) {
    //so many vars
    int n = points.size();
    vector<float> cpu_x(n);
    vector<float> cpu_y(n);
    vector<float> cpu_z(n);
    vector<int> cpu_clusters(n);
    vector<float> cpu_centroid_x(k);
    vector<float> cpu_centroid_y(k);
    vector<float> cpu_centroid_z(k);
    //convert from pointer
    for (int i = 0; i < k; i++) {
        cpu_centroid_x[i] = centroids[i].x;
        cpu_centroid_y[i] = centroids[i].y;
        cpu_centroid_z[i] = centroids[i].z;
    }
    //convert from pointer
    for (int i = 0; i < n; i++) {
        cpu_x[i] = points[i].x;
        cpu_y[i] = points[i].y;
        cpu_z[i] = points[i].z;
    }
    //GPU versions of data
    float *gpu_x;
    float *gpu_y;
    float *gpu_z;
    int *gpu_clusters;

    float *gpu_centroid_x;
    float *gpu_centroid_y;
    float *gpu_centroid_z;

    float *gpu_sum_x;
    float *gpu_sum_y;
    float *gpu_sum_z;
    int *gpu_counts;
    //allocating them
    CUDA_CHECK(cudaMalloc(&gpu_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_z, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_clusters, n * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&gpu_centroid_x, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_centroid_y, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_centroid_z, k * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&gpu_sum_x, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_sum_y, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_sum_z, k * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&gpu_counts, k * sizeof(int)));
    //memcopy
    CUDA_CHECK(cudaMemcpy(gpu_x, cpu_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_y, cpu_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_z, cpu_z.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(gpu_centroid_x, cpu_centroid_x.data(), k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_centroid_y, cpu_centroid_y.data(), k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_centroid_z, cpu_centroid_z.data(), k * sizeof(float), cudaMemcpyHostToDevice));

    //threads, blocks and share memory
    int numBlocks = (n + threadsInBlock - 1) / threadsInBlock;
    int numBlocksK= (k + threadsInBlock - 1) / threadsInBlock;
    int sharedMemSize = (3 * k * sizeof(float)) + (k * sizeof(int));
   for (int epoch = 0; epoch < epochs; epoch++) {

       assignClusters<<<numBlocks, threadsInBlock>>>(gpu_x, gpu_y, gpu_z,gpu_clusters,
       gpu_centroid_x, gpu_centroid_y, gpu_centroid_z,k,n);
       CUDA_CHECK(cudaGetLastError());

       resetArrays<<<numBlocksK, threadsInBlock>>>(gpu_sum_x, gpu_sum_y, gpu_sum_z, gpu_counts, k);
       CUDA_CHECK(cudaGetLastError());

       accumCentroids<<<numBlocks, threadsInBlock, sharedMemSize>>>(gpu_x, gpu_y, gpu_z,
           gpu_clusters,
           gpu_sum_x, gpu_sum_y, gpu_sum_z,
           gpu_counts, n,k);
       CUDA_CHECK(cudaGetLastError());

       updateCentroids<<<numBlocksK, threadsInBlock>>>(gpu_centroid_x, gpu_centroid_y, gpu_centroid_z,
           gpu_sum_x, gpu_sum_y, gpu_sum_z,
           gpu_counts, k);
       CUDA_CHECK(cudaGetLastError());
   }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(cpu_clusters.data(), gpu_clusters, n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        points[i].cluster = cpu_clusters[i];
    }
    
    CUDA_CHECK(cudaMemcpy(cpu_centroid_x.data(), gpu_centroid_x, k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_centroid_y.data(), gpu_centroid_y, k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_centroid_z.data(), gpu_centroid_z, k * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < k; i++) {
        centroids[i].x = cpu_centroid_x[i];
        centroids[i].y = cpu_centroid_y[i];
        centroids[i].z = cpu_centroid_z[i];
    }
    //makes sure to free up all the vars
    cudaFree(gpu_x);
    cudaFree(gpu_y);
    cudaFree(gpu_z);
    cudaFree(gpu_clusters);
    cudaFree(gpu_centroid_x);
    cudaFree(gpu_centroid_y);
    cudaFree(gpu_centroid_z);
    cudaFree(gpu_sum_x);
    cudaFree(gpu_sum_y);
    cudaFree(gpu_sum_z);
    cudaFree(gpu_counts);
}
