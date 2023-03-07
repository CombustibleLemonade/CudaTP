#include <iostream>
#include <dirent.h>

#include "parser/matrix_parser.h"
#include "parser/parser.h"
#include "parser/names.h"
#include "learner/structures/tree.h"
#include "prover/prover.h"



int main(int argc, char *argv[]) {
    // cudaSetDevice(1);

    std::string problem_path = "data/probabilities/";

    if (argc > 1) {
        problem_path = argv[1];
    }

    std::cout << "Problem path: " << problem_path << std::endl;

    CudaTheorem* theorem = parse_file(problem_path.c_str());
    theorem->fill_matrix();
    ComputationResult* result = start_prover(theorem);
    std::cout << "Found " << result->total_solutions << " proofs for theorem " << problem_path << std::endl;

    if (result->total_solutions > 0) {
        for (int i = 0; i < BLOCK_COUNT * THREAD_PER_BLOCK; i++) {
            if (result->solution_counts[i] > 0) {
                std::cout << "Solution of length " << result->solutions[i].length << std::endl;
                if (result->solutions[i].length == 0) continue;
                for (int j = 0; j < result->solutions[i].length; j++) {
                    SolutionStep* step = &result->solutions[i].steps[j];
                    if (step->step_type == EXTEND) std::cout << "Extend";
                    if (step->step_type == REDUCE) std::cout << "Reduce";
                    if (step->step_type == RETRACT) std::cout << "Retract";
                    std::cout << " from " << step->from_clause_idx << " " << step->from_literal_idx << " to: " << step->to_clause_idx << " " << step->to_literal_idx << std::endl;
                }
            }
        }
    }

    delete theorem;
    delete result;
}