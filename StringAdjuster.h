//
// Created by elena on 14/02/18.
//

#ifndef NAME_ENTITIES_STRINGADJUSTER_H

#define NAME_ENTITIES_STRINGADJUSTER_H
#include <iostream>
using namespace std;


class StringAdjuster {
public:
    static string removeTags(const string &s);
    static string removeNonAlphaNum(const string &s);
    static string removeMultipleSpaces(const string &s);
    static string removeFirstWhiteSpaceChar(const string &s);
    static string toLowerCase(const string &s);


};


#endif //NAME_ENTITIES_STRINGADJUSTER_H
