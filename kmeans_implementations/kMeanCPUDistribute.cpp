#include "kMeanCPUDistribute.hpp"

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ, int n){
  std::vector<Point> points;

  for(int i = 0; i < n; i++){
    Point point(vecX->at(i),vecY->at(i),vecZ->at(i));
    points.push_back(point);
  }
  return points;
}

std::vector<Point> deserializePoint(std::vector<double>* vecX,std::vector<double>* vecY,std::vector<double>* vecZ,std::vector<int>* vecCluster, int n){
  std::vector<Point> points;
  
  for(int i = 0; i < n; i++){
    Point point(vecX->at(i),vecY->at(i),vecZ->at(i));
    point.cluster = vecCluster->at(i);
    points.push_back(point);
  }
  return points;
}

std::vector<Point>* kMeansClustering(std::vector<Point>* points, std::vector<Point>& centroids, int epochs, int k,int mySize, int myRank ,int commSize){
  
  for(int i = 0; i < epochs; i++){

    for(int it = 0; it < mySize; it++){
      Point &p = points->at(it);
      for(int c = 0; c < k; c++){
        double dist = centroids[c].distance(p);
        if(dist < p.minDist){
          p.minDist = dist;
          p.cluster = c;
        }
      }
    }
    
    std::vector<int> nPoints(k, 0);
    std::vector<double> sumX(k, 0), sumY(k, 0), sumZ(k, 0);

    for(int it = 0; it < mySize; it++){
      int clusterId = points->at(it).cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] += points->at(it).x;
      sumY[clusterId] += points->at(it).y;
      sumZ[clusterId] += points->at(it).z;

      points->at(it).minDist = __DBL_MAX__;
    }

    std::vector<int> nPointsGlobal(k);
    std::vector<double> sumXGlobal(k), sumYGlobal(k), sumZGlobal(k);

    MPI_Allreduce(nPoints.data(), nPointsGlobal.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumX.data(), sumXGlobal.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumY.data(), sumYGlobal.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(sumZ.data(), sumZGlobal.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int clusterId = 0; clusterId < k; ++clusterId) {
      if (nPointsGlobal[clusterId] > 0) {
        centroids[clusterId].x = sumXGlobal[clusterId] / nPointsGlobal[clusterId];
        centroids[clusterId].y = sumYGlobal[clusterId] / nPointsGlobal[clusterId];
        centroids[clusterId].z = sumZGlobal[clusterId] / nPointsGlobal[clusterId];
      }
    }
  } 

  return points;
}

void kMeanDistribute(std::vector<Point>& points,std::vector<Point>& centroids,int epochs, int k,std::string output_dir){
  int myRank, commSize;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  
  double start = MPI_Wtime();

  int * sendCount = new int[commSize];
  int * displs = new int[commSize];
  int sum = 0;
  int n = points.size();

  std::vector<double> globalArrX;
  std::vector<double> globalArrY;
  std::vector<double> globalArrZ;

  if(myRank == 0){
    globalArrX.resize(n);
    globalArrY.resize(n);
    globalArrZ.resize(n);
    for (int i = 0; i < n; i++) {
      globalArrX[i] = points[i].x;
      globalArrY[i] = points[i].y;
      globalArrZ[i] = points[i].z;
    }
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

  std::vector<double> localArrX(sendCount[myRank]);
  std::vector<double> localArrY(sendCount[myRank]);
  std::vector<double> localArrZ(sendCount[myRank]);

  MPI_Scatterv(globalArrX.data(),sendCount,displs, MPI_DOUBLE, localArrX.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrY.data(),sendCount,displs, MPI_DOUBLE, localArrY.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrZ.data(),sendCount,displs, MPI_DOUBLE, localArrZ.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);

  std::vector<Point> pointVec = deserializePoint(&localArrX, &localArrY, &localArrZ,sendCount[myRank]);

  std::vector<Point>* localVec = kMeansClustering(&pointVec, centroids, epochs, k,sendCount[myRank],myRank,commSize);
  
  double end = MPI_Wtime();

  if(myRank == 0){
    std::cout << "Distributed CPU KMeans took " << (end - start) << " seconds." << std::endl;
  }

  std::vector<double> localKArrX(sendCount[myRank]);
  std::vector<double> localKArrY(sendCount[myRank]);
  std::vector<double> localKArrZ(sendCount[myRank]);
  std::vector<int> localKArrCluster(sendCount[myRank]);

  for(int i = 0; i < sendCount[myRank]; i++){
    localKArrX[i] = localVec->at(i).x;
    localKArrY[i] = localVec->at(i).y;
    localKArrZ[i] = localVec->at(i).z;
    localKArrCluster[i] = localVec->at(i).cluster;
  }

  int sizeVec = sendCount[myRank];
  
  std::vector<double> globalKArrX(n);
  std::vector<double> globalKArrY(n);
  std::vector<double> globalKArrZ(n);
  std::vector<int> globalKArrCluster(n);
   
  int * revIntCount = new int[commSize];
  
  MPI_Gather(&sizeVec,1, MPI_INT, revIntCount,1, MPI_INT,0, MPI_COMM_WORLD);

  int * finalDispls = NULL;

  if(myRank == 0){
    finalDispls = new int[commSize];
    finalDispls[0] = 0;
    
    for(int i = 1; i < commSize; i++){
      finalDispls[i] = finalDispls[i-1] + revIntCount[i-1];
    } 
  }
  
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
}