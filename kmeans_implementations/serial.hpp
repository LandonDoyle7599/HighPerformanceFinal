#pragma once
#include <vector>
#include "helpers.hpp"

using namespace std;
void performSerialKMeans(vector<Point>& points, int epochs, int k, vector<Point>& centroids, const string& output_dir);
void serialKMeansClustering(vector<Point>& points, int epochs, int k, vector<Point>& centroids);