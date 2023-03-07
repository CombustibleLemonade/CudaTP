#include "prover.h"

#include "../parser/parser.h"

#include <fstream>
#include <iostream>

void load_matrix_from_file(CudaTheorem* th, std::string filename){
	th->fill_matrix();
	
	CudaMatrix* m = &th->matrix;

	std::string line;
	std::ifstream file(filename);

	CudaMatrixIndex *index = nullptr; 
	int matrix_index_index = 0;

	int matrix_entry_index = 0;

	if (file.is_open()){
		int previous_clause = -2;
		int previous_literal = -2;

		while (std::getline(file, line)){
			int clause_from = -1;
			int literal_from = -1;

			int clause_to = -1;
			int literal_to = -1;

			float probability = 0.0f;
			
			int count = 0;
			int start = 0;

			while (line[++count] != ' '){}
			clause_from = std::stoi(line.substr(start, count));
			start = ++count;

			while (line[++count] != ' '){}
			literal_from = std::stoi(line.substr(start, count));
			start = ++count;
			
			while (line[++count] != ' '){}
			clause_to = std::stoi(line.substr(start, count));
			start = ++count;

			while (line[++count] != ' '){}
			literal_to = std::stoi(line.substr(start, count));
			start = ++count;

			probability = std::stof(line.substr(start, line.size()));

			if (clause_from != previous_clause || literal_from != previous_literal) {
				index = &m->indices[clause_from * 8 + literal_from];
				matrix_entry_index = index->index;

				previous_clause = clause_from;
				previous_literal = literal_from;
			}

			// assert(m->list[matrix_entry_index] == clause_to * 8 + literal_to);
			m->probabilities[matrix_entry_index] = probability;

			matrix_entry_index++;
		}
	}
}

void register_solutions(CudaTheorem* th, ComputationResult* r, std::string save_target) {
	CudaMatrix* target = new CudaMatrix;
	*target = th->matrix;

	// Set weights to 0
	for (int i = 0; i < CLAUSE_COUNT * CLAUSE_COUNT * 8; i++){
		target->probabilities[i] = 0.0f;
	}

	// Increase weights according to amount of time extension or reduction has been used in proof
	for (int i = 0; i < BLOCK_COUNT * THREAD_PER_BLOCK; i++){
		if (r->solution_counts[i] > 0) {
			Solution* solution = &r->solutions[i];

			for (int j = 0; j < solution->length; j++){
				SolutionStep step = solution->steps[j];
				if (step.step_type != CudaActionType::RETRACT){
					int entry_index = step.from_clause_idx * CLAUSE_WIDTH + step.from_literal_idx;
					CudaMatrixIndex matrix_index = target->indices[entry_index];

					bool found_match = false;

					for (int k = matrix_index.index; k < matrix_index.index + matrix_index.length; k++){
						int target_literal_encoding = target->list[k];
						int target_literal_idx = target_literal_encoding % CLAUSE_WIDTH;
						int target_clause_idx = target_literal_encoding / CLAUSE_WIDTH;

						if (target_literal_idx == step.to_literal_idx && target_clause_idx == step.to_clause_idx){
							found_match = true;
							target->probabilities[k] += 1.0;
							break;
						}
					}

					// assert(found_match);
				}
			}
		}
	}

	// Normalize weights of matrix
	for (int c = 0; c < th->clause_list.length; c++) {
        for (int l = 0; l < th->clause_list.clauses[c].length; l++) {
            CudaMatrixIndex matrix_index = target->indices[c * CLAUSE_WIDTH + l];
            float total = 0.0;

			for (int i = 0; i < matrix_index.length; i++) {
                int index = i + matrix_index.index;
                total += target->probabilities[index];
            }

            
            for (int i = 0; i < matrix_index.length; i++) {
                int index = i + matrix_index.index;
                target->probabilities[index] /= total;
            }
		}
	}

	// Write matrix to file
	std::ofstream output(save_target, std::ofstream::trunc);

	output << "# " << r->total_solutions << std::endl;

	for (int c = 0; c < th->clause_list.length; c++) {
        for (int l = 0; l < th->clause_list.clauses[c].length; l++) {
            CudaMatrixIndex matrix_index = target->indices[c * CLAUSE_WIDTH + l];
			for (int i = matrix_index.index; i < matrix_index.index + matrix_index.length; i++){
				if (!isnan(target->probabilities[i])){
					int encoded_literal = target->list[i];
					int c_to = encoded_literal / 8;
					int l_to = encoded_literal % 8;

					output << c << " " << l << " " << c_to << " " << l_to << " " << target->probabilities[i] << std::endl;
				}
			}
		}
	}
	output.close();

	delete target;
}

ComputationResult* approximate_probabilities(std::string theorem_name, std::string target_folder) {
    CudaTheorem* theorem = parse_file(("data/problems/" + theorem_name + ".th").c_str());
    
	// load_matrix_from_file(theorem, "data/weights/" + theorem_name + ".mtrx");
	theorem->fill_matrix();

    ComputationResult* result = start_prover(theorem);

    std::cout << "Found " << result->total_solutions << " proofs for theorem " << theorem_name << std::endl;
    if (result->total_solutions > 0) {
        register_solutions(theorem, result, target_folder + theorem_name + ".mtrx");
    }

    delete theorem;

    return result;
}

ComputationResult* prove_theorem_cpu(const char* path) {
    // Load the theorem
    CudaProofState* state = new CudaProofState;
    state->theorem = parse_file(path);

    // Initialize a computationresult
    ComputationResult* result = new ComputationResult;

    state->prove();
    
    return result; // TODO
}

ComputationResult* prove_theorem_gpu(const char* path) {
    // Load the theorem
    CudaTheorem* th = parse_file(path);

    // Load the theorem
    return NULL; // TODO
}