#include <fstream>
#include "Histogram.h"


void Histogram::printHistogram() {
    int max = 0;
    for (auto &it : data) {
        string s = it.first;
        if(s.length() > max) max = static_cast<int>(s.length());
    }
    for (auto &it : data) {
        string s1 = it.first;
        int len = static_cast<int>(s1.length());
        if(max - len > 0){
            for(int k=0; k < max - len; k++){
                s1.push_back(' ');
            }
        }
        string s2;
        int n = it.second;


        for(int j = 0; j < n; j++){
            s2.push_back('*');

        }


        string ss = to_string(n);
        s2.push_back('\t');
        s2.push_back('(');
        for (char s : ss) {
            s2.push_back(s);
        }
        s2.push_back(')');

        cout << s1 <<  " : " << s2 << endl;
    }
    cout << endl;
}

void Histogram::printHistogram(const int &n) {
    int max = 0;
    for(auto it =data.begin(); it != data.begin() + n; it++) {
        string s = it->first;
        if(s.length() > max) max = static_cast<int>(s.length());
    }
    for(auto it =data.begin(); it != data.begin() + n; it++) {
        string s1 = it->first;
        auto len = static_cast<int>(s1.length());
        if(max - len > 0){
            for(int k=0; k < max - len; k++){
                s1.push_back(' ');
            }
        }
        string s2;
        int nn = it->second;


        for(int j = 0; j < nn; j++){
            s2.push_back('*');

        }

        //da togliere
        string ss = to_string(nn);
        s2.push_back('\t');
        s2.push_back('(');
        for (char s : ss) {
            s2.push_back(s);
        }
        s2.push_back(')');

        cout << s1 <<  " : " << s2 << endl;
    }
    cout << endl;
}

void Histogram::saveResults(const string &path) {
    std::ofstream o(path);

    o << "{";

    auto it = data.begin();

    o << it->first << " -> " << it -> second;

    it++;

    while(it != data.end()) {
        o << " , " << it->first << " -> " << it->second;
        it++;
    }



    o << "}" << endl;

    o.close();

}

void Histogram::saveData(const string &path) {
    std::ofstream o(path);

    o << name << " found: " << to_string(data.size()) << endl;
    o << "The most frequent is " << data.begin()->first << ", which has been detected " << to_string(data.begin()->second) << " times." << endl;
    o.close();
}

void Histogram::savePartialHistogram(const string &path, const int &n) {
    std::ofstream o(path);
    int max = 0;
    for(auto it =data.begin(); it != data.begin() + n; it++) {
        string s = it->first;
        if(s.length() > max) max = static_cast<int>(s.length());
    }
    for(auto it =data.begin(); it != data.begin() + n; it++) {
        string s1 = it->first;
        auto len = static_cast<int>(s1.length());
        if(max - len > 0){
            for(int k=0; k < max - len; k++){
                s1.push_back(' ');
            }
        }
        string s2;
        int nn = it->second;


        for(int j = 0; j < nn; j++){
            s2.push_back('*');

        }


        string ss = to_string(nn);
        s2.push_back('\t');
        s2.push_back('(');
        for (char s : ss) {
            s2.push_back(s);
        }
        s2.push_back(')');

        o << s1 <<  " : " << s2 << endl;
    }


    o.close();
}