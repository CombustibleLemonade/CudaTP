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

	DIR *dir = opendir(path.c_str());
	struct dirent *diread;

	if (dir != nullptr) {
		while ((diread = readdir(dir)) != nullptr) {
            std::string name = diread->d_name;
			if (name == ".") continue;
			if (name == "..") continue;
            
            name.erase(name.size() - 5);
      
            try {
                ComputationResult* result = approximate_probabilities(name, probabilities_folder_path);
                delete result;
            } catch (...) {
                std::cout << "Exception for theorem " << name << std::endl;
            }
        }
    }
    
    // // cudaMemcpyToSymbol("a", &ah, sizeof(float *), size_t(0),cudaMemcpyHostToDevice);

    // ProbabilityMatrix matrix("data/probabilities/t82_gfacirc1.mtrx"); 

    // // CudaTheorem* th = parse_file("data/problems/l1_aff_4.th");
    // CudaTheorem* th = parse_file("data/problems/t82_gfacirc1.th");
    // std::cout << "Filling matrix" << std::endl;
    // th->fill_matrix();

    // TheoremStructure structure(th, &matrix);
    // StructureScheduler scheduler;

    // std::cout << "Adding structure" << std::endl;
    // scheduler.add_structure(&structure);


    // cudaDeviceSynchronize();

    // std::cout << "Starting training" << std::endl;
    // scheduler.start_batch();

    // std::cout << "Done" << std::endl;
}