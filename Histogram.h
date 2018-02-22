#ifndef NAME_ENTITIES_HISTOGRAM_H
#define NAME_ENTITIES_HISTOGRAM_H

#include <iostream>
#include <vector>
using namespace std;


class Histogram {
private:
    std::vector<std::pair<string,int>> data;
    string name;
public:
    Histogram(std::vector<std::pair<string,int>> d, string n){ data = d; name = n;}

    void printHistogram();

    void printHistogram(const int &n);

    void saveResults(const string &path);

    void saveData(const string &path);

    void savePartialHistogram(const string &path, const int &n);
};


#endif //NAME_ENTITIES_HISTOGRAM_H
