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
/** deserializePoint method is a overload method to deserialize a local x,y,z, and n vector to a class point vector 
 *  Input std::vector<double>* vecX
 *        std::vector<double>* vecY
 *        std::vector<double>* vecZ
          int n - number of points
 * Output std::vector<Point> 
 */
std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ, int n);
/** deserializePoint method is a overload method to deserialize a local x,y,z, vecCluster, and n vector to a class point vector 
 *  Input std::vector<double>* vecX
 *        std::vector<double>* vecY
 *        std::vector<double>* vecZ
 *        std::vector<int>* vecCluster
 *        int n - number of points
 * Output std::vector<Point> 
 */
std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster, int n);
/** deserializePoint method is a overload method to deserialize a local x,y,z, vecCluster, vecMinDist vector to a class point vector 
 *  Input std::vector<double>* vecX
 *        std::vector<double>* vecY
 *        std::vector<double>* vecZ
 *        std::vector<int>* vecCluster
 *        std::vector<int>* vecMinDist
 * Output std::vector<Point> 
 */
std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster,std::vector<int>* vecMinDist);
/**
 * kMeansClustering calcuate the kMean for the distribute CPU using the points, centroids, epochs, k, mySize, myRank, and commSize.
 * input: std::vector<Point> * centroids - random selected points
 *        std::vector<Point> * points
 *        int epochs
 *        int k
 *        int mySize
 *        int myRank
 *        int commSize
 * Output std::vector<Point> * points
 */
std::vector<Point> * kMeansClustering(std::vector<Point> * centroids,std::vector<Point> * points, int epochs, int k,int mySize, int myRank ,int commSize);
/**
 * kMeanDistributePerformance method calculates the kMean using MPI communcation. The method runs with epoch, k, filename, centroid, output_dir, points, and output to a file.
 * Input: int argc
 *        char* argv[]  
 *        int epochs
 *        int k
 *        std::string fileName
 *        std::string output_dir
 *        std::vector<Point>& centroids, 
 *        std::vector<Point>& points
 * Output file
 */
void kMeanDistributePerformance(int argc, char* argv[], int epochs, int k, std::string fileName, std::string output_dir, std::vector<Point>& centroids, std::vector<Point>& points);