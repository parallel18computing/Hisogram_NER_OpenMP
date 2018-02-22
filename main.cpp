#include <iostream>
#include <algorithm>
#include <omp.h>
#include </home/giulia/Scaricati/MITIE-master/mitielib/include/mitie/named_entity_extractor.h>
#include </home/giulia/Scaricati/MITIE-master/mitielib/include/mitie/conll_tokenizer.h>
#include </home/giulia/Scaricati/dlib-19.9.0/dlib/serialize.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "Histogram.h"
#include "StringAdjuster.h"

using namespace std;
using namespace mitie;
using namespace dlib;
using namespace std::chrono;
using std::tolower;

std::vector<string> tokenize_file (const string& filename) {
    ifstream fin(filename.c_str());
    if (!fin)
    {
        cout << "Unable to load input text file" << endl;
        exit(EXIT_FAILURE);
    }
    // The conll_tokenizer splits the contents of an istream into a bunch of words and is
    // MITIE's default tokenization method.
    conll_tokenizer tok(fin);
    std::vector<string> tokens;
    string token;
    // Read the tokens out of the file one at a time and store into tokens.

    while(tok(token)) {
        token = StringAdjuster::removeTags(token);
        //token = StringAdjuster::removeNonAlphaNum(token);
        //token = StringAdjuster::removeMultipleSpaces(token);
        //token = StringAdjuster::removeFirstWhiteSpaceChar(token);

        if(token.length() > 1) {
            tokens.push_back(token);
        }

    }

    return tokens;
}

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

struct sort_map : std::binary_function<std::pair<std::string, int>,std::pair<std::string, int>,bool> {
    inline bool operator()( const std::pair<std::string, int>& lhs, const std::pair<std::string, int>& rhs )
    {
        return lhs.second > rhs.second;
    }
};



int main() {

    int const numDataset = 3;
    int const numRepetition = 5;
    int const numDataHistogram = 10;

    int const numThread = 4;

    std::array<string,numDataset> files;
    std::array<std::vector<string>,numDataset*numRepetition> allTokens;

    for(int i=0; i < numDataset;i++){

        string file_name = "/home/giulia/CLionProjects/name_entities/tweets/dataset";
        string c = to_string(i);

        for(int j = 0; j < c.length();j++){
            file_name.push_back(c[j]);
        }

        file_name.push_back('.');
        file_name.push_back('x');
        file_name.push_back('m');
        file_name.push_back('l');

        files[i] = file_name;

    }

    string model = "/home/giulia/CLionProjects/name_entities/ner_model.dat";

    std::array<named_entity_extractor,numThread> ners;

    for(int i=0; i < numThread; i++){
        string classname;
        deserialize(model) >> classname >> ners[i];
    }


    //string classname;
    //named_entity_extractor ner;
    //deserialize(model) >> classname >> ner;

    const std::vector<string> tagstr = ners[0].get_tag_name_strings();

    cout << "The tagger supports "<< tagstr.size() - 1 <<" tags:" << endl;

    for (int i = 0; i < tagstr.size() - 1; ++i)
        cout << "   " << tagstr[i] << endl;


    for(int i = 0; i < numDataset*numRepetition; i++){

        allTokens[i] = tokenize_file(files[i%numDataset]);
    }


    std::map<string, int> results;

    omp_set_num_threads(numThread);

    system_clock::time_point ini = system_clock::now();
    time_t inizio = system_clock::to_time_t(ini);

#pragma omp parallel
    {
        std::map<string, int> local_results;
#pragma omp for schedule(static, 1)
        for (int i = 0; i < numDataset*numRepetition; i++) {
            system_clock::time_point start = system_clock::now();


            std::vector<string> tokens = allTokens[i];


            std::vector<pair<unsigned long, unsigned long> > chunks;
            std::vector<unsigned long> chunk_tags;
            std::vector<double> chunk_scores;

            //string classname2;
            //named_entity_extractor ner2;
            //deserialize(model) >> classname2 >> ner2;

            ners[omp_get_thread_num()].predict(tokens,chunks,chunk_tags,chunk_scores);

            //ner2.predict(tokens,chunks,chunk_tags,chunk_scores);

            //ner.predict(tokens, chunks, chunk_tags, chunk_scores);


            //cout << "\nNumber of named entities detected: " << chunks.size() << "\n" << endl;
            /*for (unsigned int i = 0; i < chunks.size(); ++i)
            {
                cout << "   Tag " << chunk_tags[i] << ": ";
                cout << "Score: " << fixed << setprecision(3) << chunk_scores[i] << ": ";
                cout << tagstr[chunk_tags[i]] << ": ";
                // chunks[i] defines a half open range in tokens that contains the entity.
                for (unsigned long j = chunks[i].first; j < chunks[i].second; ++j)
                    cout << tokens[j] << " ";
                cout << endl;
            }*/

            for (int ii = 0; ii < chunks.size(); ii++) {
                string tag = tagstr[chunk_tags[ii]];
                string name = "";
                for (long j = chunks[ii].first; j < chunks[ii].second; j++) {
                    name += tokens[j] + " ";
                }

                name = StringAdjuster::toLowerCase(name);
                name = StringAdjuster::removeNonAlphaNum(name);
                name = StringAdjuster::removeMultipleSpaces(name);
                name = StringAdjuster::removeFirstWhiteSpaceChar(name);


                string key = name + "_" + tag;


                if (local_results.find(key) != local_results.end()) {
                    int val = local_results.at(key);
                    local_results.at(key) = ++val;

                } else {
                    local_results.insert(std::pair<string, int>(key, 1));
                }


            }

            system_clock::time_point end = system_clock::now();


            printf("thread %d : token %d , start: %d duration: %d \n",omp_get_thread_num(),i, chrono::duration_cast<chrono::milliseconds>(start - ini).count(), chrono::duration_cast<chrono::milliseconds>(end - start).count()  );

        }

#pragma omp critical
        {
            for(map<string, int >::const_iterator it = local_results.begin(); it != local_results.end(); ++it) {
                string key = it->first;
                int local_val = it-> second;

                if (results.find(key) != results.end()) {
                    int global_val = results.at(key);
                    int total_val  = global_val + local_val;
                    results.at(key) = total_val;

                } else {
                    results.insert(std::pair<string, int>(key, local_val));
                }


            }
        }

    }




    system_clock::time_point fin = system_clock::now();

    //fin = system_clock::now();

    //time_t fine = system_clock::to_time_t(fin);

    //time_t tempo = fine - inizio;

    /*for(map<string, int >::const_iterator it = results.begin(); it != results.end(); ++it)
    {
        cout << it->first << " -> " << it->second << endl;
    }*/


    std::map<string,int> persons;
    std::map<string,int> locations;
    std::map<string,int> organizations;

    for(map<string, int >::const_iterator it = results.begin(); it != results.end(); ++it)
    {
        string key = it->first;
        int val = it->second;

        string p = "_PERSON";
        string l = "_LOCATION";
        string o = "_ORGANIZATION";

        if(hasEnding(key,p)){
            key.replace(key.end() - p.size(),key.end(),"");
            persons.insert(std::pair<string,int>(key,val));
        }else
        if(hasEnding(key,l)){
            key.replace(key.end() - l.size(),key.end(),"");
            locations.insert(std::pair<string,int>(key,val));
        }else
        if(hasEnding(key,o)){
            key.replace(key.end() - o.size(),key.end(),"");
            organizations.insert(std::pair<string,int>(key,val));
        }


    }

    std::vector<std::pair<string,int>> sorted_persons(persons.begin(),persons.end());
    std::sort(sorted_persons.begin(),sorted_persons.end(),sort_map());

    std::vector<std::pair<string,int>> sorted_locations(locations.begin(),locations.end());
    std::sort(sorted_locations.begin(),sorted_locations.end(),sort_map());

    std::vector<std::pair<string,int>> sorted_organizations(organizations.begin(),organizations.end());
    std::sort(sorted_organizations.begin(),sorted_organizations.end(),sort_map());

    cout<<endl;


    cout << "PERSONS: " << sorted_persons.size() << endl;
    Histogram * hPersons = new Histogram(sorted_persons, "Persons");
    hPersons->printHistogram(numDataHistogram);
    hPersons->saveResults("/home/giulia/CLionProjects/name_entities_openMP/results/persons/all_persons.txt");
    hPersons->saveData("/home/giulia/CLionProjects/name_entities_openMP/results/persons/data_persons.txt");
    hPersons->savePartialHistogram("/home/giulia/CLionProjects/name_entities_openMP/results/persons/partial_histogram_persons.txt", numDataHistogram);

    cout << "LOCATIONS: " << sorted_locations.size() << endl;
    Histogram * hLocations = new Histogram(sorted_locations, "LOCATIONS");
    hLocations->printHistogram(numDataHistogram);
    hLocations->saveResults("/home/giulia/CLionProjects/name_entities_openMP/results/locations/all_locations.txt");
    hLocations->saveData("/home/giulia/CLionProjects/name_entities_openMP/results/locations/data_locations.txt");
    hLocations->savePartialHistogram("/home/giulia/CLionProjects/name_entities_openMP/results/locations/partial_histogram_locations.txt", numDataHistogram);

    cout << "ORGANIZATIONS: " << sorted_organizations.size() << endl;
    Histogram * hOrganizations = new Histogram(sorted_organizations,"ORGANIZATION");
    hOrganizations->printHistogram(numDataHistogram);
    hOrganizations->saveResults("/home/giulia/CLionProjects/name_entities_openMP/results/organizations/all_organizations.txt");
    hOrganizations->saveData("/home/giulia/CLionProjects/name_entities_openMP/results/organizations/data_organizations.txt");
    hOrganizations->savePartialHistogram("/home/giulia/CLionProjects/name_entities_openMP/results/organizations/partial_histogram_organizations.txt", numDataHistogram);

    cout << "Execution time: " << chrono::duration_cast<chrono::milliseconds>(fin - ini).count() << endl;


    return 0;
}
