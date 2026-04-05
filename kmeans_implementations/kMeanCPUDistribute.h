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

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster,std::vector<int>* vecMinDist);

bool isNumber(std::string s);


std::tuple<double,double,double> readcsvPoint(std::string csv);

std::vector<Point> * kMeansClustering(std::vector<Point> * points, int epochs, int k,int mySize, int myRank ,int commSize);


void writeFile(std::string fileName,std::vector<Point>*points);

void kMeanDistributePerformance(int argc, char*argv[],int epochs, int k, std::string fileName);
void kMeanDistribute(int argc, char*argv[],int epochs, int k, std::string fileName);
