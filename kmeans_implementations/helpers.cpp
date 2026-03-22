#include "helpers.hpp"
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

Point::Point() : x(0), y(0), z(0), cluster(-1), minDist(DBL_MAX) {}
Point::Point(double x, double y, double z) : x(x), y(y), z(z), cluster(-1), minDist(DBL_MAX) {}
double Point::distance(const Point& p) const {
    return (x - p.x)*(x - p.x) +
           (y - p.y)*(y - p.y) +
           (z - p.z)*(z - p.z);
}


void writeOutput(vector<Point>& points, const string& filename) {
    ofstream file(filename);
    file << "energy,speechiness,liveness,cluster" << endl;
    for (const auto& p: points) {
        file << p.x << "," << p.y << "," << p.z << "," << p.cluster << endl;
    }
    file.close();
}

vector<Point> readData(const string& filename) {
    vector<Point> points;
    ifstream file(filename);
    string line;
    //skip header of csv
    getline(file, line);
    
    while (getline(file, line)) {
        stringstream lineStream(line);
        string x, y, z;
        getline(lineStream, x, ',');
        getline(lineStream, y, ',');
        getline(lineStream, z, ',');
        
        points.emplace_back(stod(x), stod(y), stod(z));
    }
    file.close();
    return points;
}