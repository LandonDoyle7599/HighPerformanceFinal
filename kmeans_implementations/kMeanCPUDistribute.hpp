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

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ, int n);

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster, int n);

std::vector<Point>* kMeansClustering(std::vector<Point>* points, std::vector<Point>& centroids, int epochs, int k,int mySize, int myRank ,int commSize);

void kMeanDistribute(std::vector<Point>& points,std::vector<Point>& centroids,int epochs, int k,std::string output_dir);