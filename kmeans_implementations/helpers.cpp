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

Args::Args(int argc, char *argv[]) {
    if (argc >= 4) {
        k = atoi(argv[1]);
        epochs = atoi(argv[2]);
        input_file = argv[3];
        output_dir = argv[4];
        for (int i = 5; i < argc; ++i) {
            string flag = argv[i];
            
            //handle skipping serial
            if (flag == "--skip_serial") skip_serial = true;

            //handle shared cpu and num threads
            else if (flag == "--shared_cpu") {
                shared_cpu = true;
                if (i + 1 < argc) {
                    num_threads = atoi(argv[i + 1]);
                    i++;
                }
            }
            
            else if (flag == "--cuda_gpu") cuda_gpu = true; //add onto this for any additional args
            else if (flag == "--dist_cpu") dist_cpu = true; //add onto this for any additional args
            else if (flag == "--dist_gpu") dist_gpu = true; //add onto this for any additional args
        }
    } else {
        throw invalid_argument("Usage: <number_of_clusters> <input_file> <output_dir> [--shared_cpu <num_threads>] [--cuda_gpu] [--dist_cpu] [--dist_gpu] [--skip_serial]");
    }
}

void writeOutput(vector<Point>& points, const string& filename) {
    std::string buffer;
    buffer.reserve(points.size() * 50);

    buffer += "x,y,z,c\n";

    for (const auto& p : points) {
        buffer += std::to_string(p.x) + "," +
                std::to_string(p.y) + "," +
                std::to_string(p.z) + "," +
                std::to_string(p.cluster) + "\n";
    }
    std::ofstream outfile(filename);
    outfile.write(buffer.c_str(), buffer.size());
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
