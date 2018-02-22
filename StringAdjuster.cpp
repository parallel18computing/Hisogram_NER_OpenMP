//
// Created by elena on 14/02/18.
//

#include <algorithm>
#include "StringAdjuster.h"


string StringAdjuster::removeTags(const string &s) {

    string text = s;

    string tag1 = "<TweetText>";
    string tag2 = "</TweetText>";
    size_t f1 = text.find(tag1);
    size_t f2 = text.find(tag2);

    if(f1 < text.length())
        text.replace(f1,tag1.length(),"");

    if(f2 < text.length())
        text.replace(f2,tag2.length(),"");

    return text;

}

string StringAdjuster::removeNonAlphaNum(const string &s) {
    string text = s;

    for(int i=0;i < text.length(); i++){
        if(!isalnum(text[i])) {
            text.replace(i,1," ");
        }
    }

    return text;
}


string StringAdjuster::removeMultipleSpaces(const string &s) {

    string text = "";

    int inputIndex = 0;

    while (inputIndex < s.length()) {

        text.push_back(s[inputIndex]);

        if (s[inputIndex] == ' ') {
            while (s[inputIndex + 1] == ' ') {
                // skip over any extra spaces
                inputIndex++;
            }
        }

        inputIndex++;
    }

    // null-terminate output
    text.push_back('\0');


    return text;



}

string StringAdjuster::removeFirstWhiteSpaceChar(const string &s)  {
    string text = s;
    if(text[0] == ' ')
        text.erase(text.begin(),text.begin() + 1);

    return text;
}


string StringAdjuster::toLowerCase(const string &s) {

    string text = s;

    for(int i=0; i < text.length(); i++) {
        text[i] = tolower(text[i]);
    }

    return text;
}