#ifndef PROVER
#define PROVER

#include <string>
#include "cudaProver.h"

void load_matrix_from_file(CudaTheorem* th, std::string filename);

ComputationResult* approximate_probabilities(std::string theorem_name, std::string target_folder="data/probabilities/");

ComputationResult* prove_theorem_cpu(const char* path);

ComputationResult* prove_theorem_gpu(const char* path);

#endif