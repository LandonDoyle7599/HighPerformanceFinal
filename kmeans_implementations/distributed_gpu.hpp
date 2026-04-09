#ifndef DISTRIBUTED_GPU_HPP
#define DISTRIBUTED_GPU_HPP

#include <vector>
#include <string>
#include "helpers.hpp"

void performDistributedGPUKMeans(
    std::vector<Point>& points,
    int epochs,
    int k,
    std::vector<Point>& centroids,
    const std::string& output_dir,
    int threadsPerBlock
);

#endif
