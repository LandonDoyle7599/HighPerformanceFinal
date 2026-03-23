#pragma once
#include <vector>
#include "helpers.hpp"

using namespace std;
void performSharedCPUKMeans(vector<Point>& points, int epochs, int k, vector<Point>& centroids, const string& output_dir, int num_threads);
void sharedCPUKMeansClustering(vector<Point>& points, int epochs, int k, vector<Point>& centroids);