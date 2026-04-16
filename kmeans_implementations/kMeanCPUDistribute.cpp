#include "kMeanCPUDistribute.hpp"

/** deserializePoint method is a overload method to deserialize a local x,y,z, and n vector to a class point vector 
 *  Input std::vector<double>* vecX
 *        std::vector<double>* vecY
 *        std::vector<double>* vecZ
          int n - number of points
 * Output std::vector<Point> 
 */
std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ, int n){
  std::vector<Point> points;

  for(int i = 0; i < n; i++){
    Point point(vecX->at(i),vecY->at(i),vecZ->at(i));
    points.push_back(point);
  }
  return points;
}
/** deserializePoint method is a overload method to deserialize a local x,y,z, vecCluster, and n vector to a class point vector 
 *  Input std::vector<double>* vecX
 *        std::vector<double>* vecY
 *        std::vector<double>* vecZ
 *        std::vector<int>* vecCluster
 *        int n - number of points
 * Output std::vector<Point> 
 */
std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster, int n){
  std::vector<Point> points;
  
  for(int i = 0; i < n; i++){
    Point point(vecX->at(i),vecY->at(i),vecZ->at(i));
    point.cluster = vecCluster->at(i);
    points.push_back(point);
  }
  return points;
}
/** deserializePoint method is a overload method to deserialize a local x,y,z, vecCluster, vecMinDist vector to a class point vector 
 *  Input std::vector<double>* vecX
 *        std::vector<double>* vecY
 *        std::vector<double>* vecZ
 *        std::vector<int>* vecCluster
 *        std::vector<int>* vecMinDist
 * Output std::vector<Point> 
 */
std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster,std::vector<int>* vecMinDist){
  std::vector<Point> points;
  int n = vecX->size();
  for(int i = 0; i < n; i++){
    Point point(vecX->at(i),vecY->at(i),vecZ->at(i));
    point.cluster = vecCluster->at(i);
    point.minDist = vecMinDist->at(i);
    points.push_back(point);
  }
  return points;
}



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
std::vector<Point> * kMeansClustering(std::vector<Point> * centroids,std::vector<Point> * points, int epochs, int k,int mySize, int myRank ,int commSize){

  
  for(int i = 0; i < epochs; i++){

    for(std::vector<Point>::iterator c = begin(*centroids); c != end(*centroids); ++c){
      int clusterId = c - begin(*centroids);
      for(int it = 0; it < mySize; it++){
        Point p = points->at(it);
        double dist = c->distance(p);
        if(dist < p.minDist){
          p.minDist = dist;
          p.cluster = clusterId;
        }
        points->at(it) = p;
      }
    }
    
    std::vector<int> nPoints;
    std::vector<double> sumX, sumY, sumZ;

    for(int j = 0; j < k; ++j) {
      nPoints.push_back(0);
      sumX.push_back(0);
      sumY.push_back(0);
      sumZ.push_back(0);
    }
    for(int it = 0; it < mySize; it++){
      int clusterId = points->at(it).cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] += points->at(it).x;
      sumY[clusterId] += points->at(it).y;
      sumZ[clusterId] += points->at(it).z;

      points->at(it).minDist = __DBL_MAX__;
    }

    std::vector<int> globalNPoints(k);
    std::vector<double> globalSumX(k), globalSumY(k), globalSumZ(k);
    // Calculate the global sum
    MPI_Allreduce(nPoints.data(), globalNPoints.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumX.data(), globalSumX.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumY.data(), globalSumY.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumZ.data(), globalSumZ.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    for (std::vector<Point>::iterator c = begin(*centroids); c != end(*centroids); ++c) {
      int clusterId = c - std::begin(*centroids);
      c->x = globalSumX[clusterId] / globalNPoints[clusterId];
      c->y = globalSumY[clusterId] / globalNPoints[clusterId];
      c->z = globalSumZ[clusterId] / globalNPoints[clusterId];
    }
            
  } 
  return points;
}
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
void kMeanDistributePerformance(int argc, char* argv[], int epochs, int k, std::string fileName, std::string output_dir, std::vector<Point>& centroids, std::vector<Point>& points){  
  double start, finish;
  int myRank, commSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  
  std::string line;
  std::ifstream file(fileName);
  
  int * sendCount = new int[commSize];
  int * displs = new int[commSize];
  int sum = 0;
  int n = 0;
  std::vector<Point> localPointArr;
  std::vector<double> globalArrX;
  std::vector<double> globalArrY;
  std::vector<double> globalArrZ;

  std::vector<double> globalCX(k, 0);
  std::vector<double> globalCY(k, 0);
  std::vector<double> globalCZ(k, 0);

  // Rank zero create the global arr and centroids/  
  if(myRank == 0){
    n = points.size();
    for (int i = 0; i < n; i++) {
        globalArrX.push_back(points[i].x);
        globalArrY.push_back(points[i].y);
        globalArrZ.push_back(points[i].z);
    }
    for (int i = 0; i < k; i++) {
        globalCX[i] = centroids[i].x;
        globalCY[i] = centroids[i].y;
        globalCZ[i] = centroids[i].z;
    }
  }
  // Rank zero broadcasts the array to other ranks.
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(globalCX.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(globalCY.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(globalCZ.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int reminder = n%commSize;
  for(int i = 0; i < commSize; i++){
    sendCount[i] = n/commSize;
    if(reminder > 0){
      sendCount[i]++;
      reminder--;
    }
    displs[i] = sum;
    sum += sendCount[i];
  }
  std::vector<double> localArrX(n,0);
  std::vector<double> localArrY(n,0);
  std::vector<double> localArrZ(n,0);

 // Scatter the globalArray.
  MPI_Scatterv(globalArrX.data(),sendCount,displs, MPI_DOUBLE, localArrX.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrY.data(),sendCount,displs, MPI_DOUBLE, localArrY.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrZ.data(),sendCount,displs, MPI_DOUBLE, localArrZ.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  // Create the global centroids.
  std::vector<Point> globalC = deserializePoint(&globalCX, &globalCY,&globalCZ, k);
  // create the local points
  std::vector<Point> pointVec = deserializePoint(&localArrX, &localArrY, &localArrZ,sendCount[myRank]);
  start = MPI_Wtime();
  std::vector<Point> * localVec = kMeansClustering(&globalC,&pointVec, epochs, k,sendCount[myRank],myRank,commSize);
    MPI_Barrier(MPI_COMM_WORLD);
  finish = MPI_Wtime();
  if(myRank == 0)
    std::cout << "DistCPUKMeans clustering with " << k << " clusters and " << commSize << " nodes took " << finish - start << " seconds." << std::endl;
  std::vector<double> localKArrX;
  std::vector<double> localKArrY;
  std::vector<double> localKArrZ;
  std::vector<int> localKArrCluster;

  // Create the local arr
  for(int i = 0; i < sendCount[myRank]; i++){
    localKArrX.push_back(localVec->at(i).x);
    localKArrY.push_back(localVec->at(i).y);
    localKArrZ.push_back(localVec->at(i).z);
    localKArrCluster.push_back(localVec->at(i).cluster);
  }
  int sizeVec = sendCount[myRank];
  
  std::vector<double> globalKArrX(n,0);
  std::vector<double> globalKArrY(n,0);
  std::vector<double> globalKArrZ(n,0);
  std::vector<int> globalKArrCluster(n,0);
   
  int * revIntCount = new int[commSize];
  
  // Gather how many points will the rank will send.
  MPI_Gather(&sizeVec,1, MPI_INT, revIntCount,1, MPI_INT,0, MPI_COMM_WORLD);
  int * finalDispls = NULL;

  if(myRank == 0){
    finalDispls = new int[commSize];
    finalDispls[0] = 0;
    
    for(int i = 1; i < commSize; i++){
      finalDispls[i] = finalDispls[i-1] + revIntCount[i-1];
    } 
  }
  // Create the global array
  MPI_Gatherv(localKArrX.data(),sizeVec,MPI_DOUBLE, globalKArrX.data(), revIntCount, finalDispls,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gatherv(localKArrY.data(),sizeVec,MPI_DOUBLE, globalKArrY.data(), revIntCount, finalDispls,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gatherv(localKArrZ.data(),sizeVec,MPI_DOUBLE, globalKArrZ.data(), revIntCount, finalDispls,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Gatherv(localKArrCluster.data(),sizeVec,MPI_INT, globalKArrCluster.data(), revIntCount, finalDispls,MPI_INT,0,MPI_COMM_WORLD);

  if(myRank == 0){
    std::vector<Point> rankZeroPoints =  deserializePoint(&globalKArrX,&globalKArrY,&globalKArrZ,&globalKArrCluster, n);
    writeOutput(rankZeroPoints, output_dir + "/dist_cpu_output.csv");
  }  
  
  delete[] sendCount;
  delete[] displs;
  delete[] revIntCount;
  delete[] finalDispls;
  return;
}