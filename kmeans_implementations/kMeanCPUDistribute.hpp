#pragma once
#include "helpers.hpp"
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cctype>
#include <mpi.h>
#include <string>
#include <tuple>
#include <algorithm>

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ, int n);

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster, int n);

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster,std::vector<int>* vecMinDist);

std::vector<Point> * kMeansClustering(std::vector<Point> * centroids,std::vector<Point> * points, int epochs, int k,int mySize, int myRank ,int commSize);

void kMeanDistributePerformance(int argc, char* argv[], int epochs, int k, std::string fileName, std::string output_dir, std::vector<Point>& centroids, std::vector<Point>& points);