#ifndef SHARED_GPU_HPP
#define SHARED_GPU_HPP

#include <vector>
#include <string>
#include "helpers.hpp"


__global__ void assignClusters(
    float* point_x, float* point_y, float* point_z,
    int* cluster,
    float* centroid_x, float* centroid_y, float* centroid_z,
    int k, int n
);

__global__ void resetArrays(
    float* sumX, float* sumY, float* sumZ,
    int* counts, int k
);

__global__ void accumCentroids(
    float* x, float* y, float* z,
    int* clusters,
    float* sumX, float* sumY, float* sumZ,
    int* counts,
    int n, int k
);
void performSharedGPUKMeans(
    std::vector<Point>& points,
    int epochs,
    int k,
    std::vector<Point>& centroids,
    const std::string& output_dir
);
void performGPUKmeans(
    std::vector<Point>& points,
    int k,
    int epochs,
    std::vector<Point>& centroids
);

#endif
