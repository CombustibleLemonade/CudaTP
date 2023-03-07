#ifndef MATRIX_PARSER
#define MATRIX_PARSER

#include <string>
#include <map>
#include <set>
#include <vector>


class ProbabilityMatrix {
public:
    std::map<std::vector<int>, float> weights;
    std::set<std::pair<int, int>> literals;

    ProbabilityMatrix(std::string filename);
};

#endif