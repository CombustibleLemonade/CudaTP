#include <iostream>
#include <dirent.h>

#include "parser/matrix_parser.h"
#include "parser/parser.h"
#include "parser/names.h"
#include "learner/structures/tree.h"
#include "prover/prover.h"

int main(int argc, char *argv[]) {
    // cudaSetDevice(1);
    std::string probabilities_folder_path = "data/probabilities/";

    if (argc > 1) {
        probabilities_folder_path = argv[1];
        std::cout << probabilities_folder_path << std::endl;
    }
    // return 0;

	std::string path = "data/weights/";

    if (argc > 2) {
        path = argv[2];
    }

	DIR *dir = opendir(path.c_str());
	struct dirent *diread;

	if (dir != nullptr) {
		while ((diread = readdir(dir)) != nullptr) {
            std::string name = diread->d_name;
			if (name == ".") continue;
			if (name == "..") continue;
            
            if (name.find(".th")) {
                name.erase(name.size() - 3);
            } else {
                name.erase(name.size() - 5);
            }
            
            try {
                ComputationResult* result = approximate_probabilities(name, probabilities_folder_path);
                delete result;
            } catch (...) {
                std::cout << "Exception for theorem " << name << std::endl;
            }
        }
    }
}