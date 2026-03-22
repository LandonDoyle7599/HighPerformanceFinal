#include <cfloat>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

struct Point {
    double x, y;
    int cluster;
    double minDist;

    Point():
        x(0), y(0), cluster(-1), minDist(DBL_MAX) {}

    Point(double x, double y):
        x(x), y(y), cluster(-1), minDist(DBL_MAX) {}

    double distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};

vector<Point> readData(string filename) {
    vector<Point> points;
    string line;
    ifstream file(filename);
    
    while (getline(file, line)) {
        stringstream lineStream(line);
        string bit;
        double x, y;
        getline(lineStream, bit, ',');
        x = stof(bit);
        getline(lineStream, bit, '\n');
        y = stof(bit);
        points.push_back(Point(x, y));
    }
    return points;
}

void kMeansClustering(vector<Point>* points, int epochs, int k){
    vector<Point> centroids;
    srand(time(0));
    for (int i = 0; i < k; ++i) {
        centroids.push_back(points->at(rand() % points->size()));
    }

    for (vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
        int clusterId = c - begin(centroids);
        for (vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
            Point p = *it;
            double dist = c->distance(p);
            if (dist < p.minDist) {
                p.minDist = dist;
                p.cluster = clusterId;
            }
            *it = p;
        }
    }

    vector<int> nPoints;
    vector<double> sumX, sumY;
    for (int j = 0; j < k; ++j) {
        nPoints.push_back(0);
        sumX.push_back(0.0);
        sumY.push_back(0.0);
    }

    for (vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
        int clusterId = it->cluster;
        nPoints[clusterId] += 1;
        sumX[clusterId] += it->x;
        sumY[clusterId] += it->y;

        it->minDist = DBL_MAX;
    }

    for (vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
        int clusterId = c - begin(centroids);
        c->x = sumX[clusterId] / nPoints[clusterId];
        c->y = sumY[clusterId] / nPoints[clusterId];
    }

    ofstream myfile;
    myfile.open("serial_output.csv");
    myfile << "x,y,c" << endl;
    for (vector<Point>::iterator it = points->begin(); it != points->end(); ++it) {
        myfile << it->x << "," << it->y << "," << it->cluster << endl;
    }
    myfile.close();
}

int main(int argc, char const *argv[])
{
    vector<Point> points = readData("trimmed_tracks_features.csv");
    kMeansClustering(&points, 5, 3);
    return 0;
}



