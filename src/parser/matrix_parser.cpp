#include "matrix_parser.h"

#include <iostream>
#include <fstream>

ProbabilityMatrix::ProbabilityMatrix(std::string filename) {
    std::ifstream file_stream(filename);
    std::string line;

    while (std::getline(file_stream, line)) {
        std::vector<int> lit_indices;
        
        int start_pos = 0;
        int delimiter_pos = 0;
        for (int i = 0; i < 4; i++) {
            while (line[delimiter_pos] != ' ') {
                delimiter_pos++;
            }
            std::string sub = line.substr(start_pos, delimiter_pos - start_pos);
            lit_indices.push_back(stoi(sub));

            delimiter_pos++;
            start_pos = delimiter_pos;
        }

        float probability = stof(line.substr(delimiter_pos));

        weights[lit_indices] = probability;
        literals.insert(std::make_pair(lit_indices[0], lit_indices[1]));
        literals.insert(std::make_pair(lit_indices[2], lit_indices[3]));
    }
}