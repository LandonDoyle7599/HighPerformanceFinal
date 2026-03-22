#pragma once
#include <vector>
#include <string>

struct Point {
    double x, y, z;
    int cluster;
    double minDist;

    Point();
    Point(double x, double y, double z);

    double distance(const Point& p) const;
};

std::vector<Point> readData(const std::string& filename);
void writeOutput(std::vector<Point>& points, const std::string& filename);