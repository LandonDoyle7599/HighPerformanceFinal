#ifndef SHARED_GPU_HPP
#define SHARED_GPU_HPP

#include <vector>
#include <string>
#include "helpers.hpp"

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
