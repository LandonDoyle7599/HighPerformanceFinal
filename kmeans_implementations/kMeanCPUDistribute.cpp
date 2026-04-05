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

bool isNumber(std::string s){
  if(s == "")
    return false;
  for(auto& c : s){
    if(!std::isdigit(c) && c != '.' && c !='-' && c !='E')
      return false;
  }
  return true;
}

std::tuple<double,double,double> readcsvPoint(std::string csv){
  std::stringstream lineStream(csv);
    int i = 0;
    std::string bit;
    double x,y,z;

    while(std::getline(lineStream, bit,',')){
      if(isNumber(bit)){
	switch(i){
	case 0:

	  x = std::stof(bit);
	  break;
	case 1:

	  y = std::stof(bit);
	  break;
	case 2:

	  z = std::stof(bit);
	  break;
	}
	i++;     
      }
    }
    return std::make_tuple(x,y,z);
}

std::vector<Point> * kMeansClustering(std::vector<Point> * points, int epochs, int k,int mySize, int myRank ,int commSize){
  std::vector<Point> centroids;
  srand(time(0));
  for(int i = 0; i < k; i++){
    centroids.push_back(points->at(rand() % points->size()));
  }
  
  for(int i = 0; i < epochs; i++){

    for(std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c){
      int clusterId = c - begin(centroids);
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
    
    for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
      int clusterId = c - std::begin(centroids);
      c->x = sumX[clusterId] / nPoints[clusterId];
      c->y = sumY[clusterId] / nPoints[clusterId];
      c->z = sumZ[clusterId] / nPoints[clusterId];
    }
     
    std::vector<double> localArrX;
    std::vector<double> localArrY;
    std::vector<double> localArrZ;
    std::vector<int> localArrCluster;
    
   
    for(int i = 0; i < k; i++){
      localArrX.push_back(centroids[i].x);
      localArrY.push_back(centroids[i].y);
      localArrZ.push_back(centroids[i].z);
      localArrCluster.push_back(centroids[i].cluster);
    }
   
    std::vector<double> localRevArrX(k);
    std::vector<double> localRevArrY(k);
    std::vector<double> localRevArrZ(k);
    std::vector<int> localRevArrCluster(k);
    
    int mySend = (myRank+1)%commSize;
    int partner = (myRank-1+commSize)%commSize;
    MPI_Sendrecv(localArrX.data(),k,MPI_DOUBLE,mySend,0,localRevArrX.data(),k,MPI_DOUBLE,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Sendrecv(localArrY.data(),k,MPI_DOUBLE,mySend,0,localRevArrY.data(),k,MPI_DOUBLE,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Sendrecv(localArrZ.data(),k,MPI_DOUBLE,mySend,0,localRevArrZ.data(),k,MPI_DOUBLE,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Sendrecv(localArrCluster.data(),k,MPI_INT,mySend,0,localRevArrCluster.data(),k,MPI_INT,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    centroids = deserializePoint(&localRevArrX, &localRevArrY, &localRevArrZ,&localRevArrCluster,k);
       
  } 
  return points;
}
void writeFile(std::string fileName,std::vector<Point>*points){
  std::ofstream myfile;
  myfile.open(fileName);
  myfile << "x,y,z,c" << std::endl;
  
  for (std::vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
    myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << std::endl;
  }
  myfile.close();
}

void kMeanDistributePerformance(int argc, char*argv[],int epochs, int k, std::string fileName){
  double start, finish;
    int myRank, commSize;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  
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
  std::getline(file,line);
  if(myRank == 0){
    while(std::getline(file,line)){
      auto result = readcsvPoint(line);
      globalArrX.push_back(std::get<0>(result));
      globalArrY.push_back(std::get<1>(result));
      globalArrZ.push_back(std::get<2>(result)); 
    }
      n = globalArrX.size();
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
  std::vector<double> localArrX(n,0);
  std::vector<double> localArrY(n,0);
  std::vector<double> localArrZ(n,0);

 
  MPI_Scatterv(globalArrX.data(),sendCount,displs, MPI_DOUBLE, localArrX.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrY.data(),sendCount,displs, MPI_DOUBLE, localArrY.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrZ.data(),sendCount,displs, MPI_DOUBLE, localArrZ.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  start = MPI_Wtime();
  std::vector<Point> pointVec = deserializePoint(&localArrX, &localArrY, &localArrZ,sendCount[myRank]);
  MPI_Barrier(MPI_COMM_WORLD);
  finish = MPI_Wtime();
  if(myRank == 0)
    printf("Elasped time = %e secounds", finish-start);
  std::vector<Point> * localVec = kMeansClustering(&pointVec, epochs, k,sendCount[myRank],myRank,commSize);
  
  std::vector<double> localKArrX;
  std::vector<double> localKArrY;
  std::vector<double> localKArrZ;
  std::vector<int> localKArrCluster;
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
    
    writeFile("output.csv", &rankZeroPoints);
  }  
  MPI_Finalize();
  
  delete[] sendCount;
  delete[] displs;
  delete[] revIntCount;
  delete[] finalDispls;

}

void kMeanDistribute(int argc, char*argv[],int epochs, int k, std::string fileName){
  int myRank, commSize;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
  
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
  std::getline(file,line);
  if(myRank == 0){
    while(std::getline(file,line)){
      auto result = readcsvPoint(line);
      globalArrX.push_back(std::get<0>(result));
      globalArrY.push_back(std::get<1>(result));
      globalArrZ.push_back(std::get<2>(result)); 
    }
      n = globalArrX.size();
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
  std::vector<double> localArrX(n,0);
  std::vector<double> localArrY(n,0);
  std::vector<double> localArrZ(n,0);

 
  MPI_Scatterv(globalArrX.data(),sendCount,displs, MPI_DOUBLE, localArrX.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrY.data(),sendCount,displs, MPI_DOUBLE, localArrY.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);
  MPI_Scatterv(globalArrZ.data(),sendCount,displs, MPI_DOUBLE, localArrZ.data() , sendCount[myRank], MPI_DOUBLE,0, MPI_COMM_WORLD);

  std::vector<Point> pointVec = deserializePoint(&localArrX, &localArrY, &localArrZ,sendCount[myRank]);
  std::vector<Point> * localVec = kMeansClustering(&pointVec, epochs, k,sendCount[myRank],myRank,commSize);
  
  std::vector<double> localKArrX;
  std::vector<double> localKArrY;
  std::vector<double> localKArrZ;
  std::vector<int> localKArrCluster;
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
    
    writeFile("output.csv", &rankZeroPoints);
  }  
  MPI_Finalize();
  
  delete[] sendCount;
  delete[] displs;
  delete[] revIntCount;
  delete[] finalDispls;

}
