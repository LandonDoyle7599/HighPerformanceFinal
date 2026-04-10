#pragma once
#include <vector>
#include <string>

struct Point {
    double x, y, z;
    int cluster;
    double minDist;

    Point();
    Point(double x, double y, double z);

    double distance(const Point& p) const;
};

struct Args {
    int k;
    std::string input_file;
    std::string output_dir;
    bool shared_cpu = false;
    int num_threads = 1;
    int threadsPerBlockDist = 256;
    int threadsPerBlockCuda = 256;
    bool cuda_gpu = false;
    bool dist_cpu = false;
    bool dist_gpu = false;
    bool skip_serial = false;

    Args(int argc, char const *argv[]);
};

std::vector<Point> readData(const std::string& filename);
void writeOutput(std::vector<Point>& points, const std::string& filename);