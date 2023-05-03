#include <iostream>
#include <thread>

#include "benchmark.h"
#include "parser/parser.h"
#include "prover/prover.h"

#define NUM_THREADS 128

void run_benchmark(std::string path) {
    // std::cout << path << std::endl;
    CudaTheorem* theorem = parse_file(("data/problems/" + path + ".th").c_str());
    load_matrix_from_file(theorem, "data/weights/" + path + ".mtrx");

    CudaProofState state;
    state.theorem = theorem;

    for (int i = 0; i < 8192; i++) {
        state.prove();

        while (state.mutation.log.action_length != 0) {
            state.undo();
        }
    }
    
    delete theorem;
}

int main(int argc, char *argv[]) {
    uint64_t start = time_since_epoch_milliseconds();
    uint64_t previous = time_since_epoch_milliseconds();
    std::cout << "Start: " << start << std::endl;

    for (int i = 0; i < problems.size(); i++) {
        std::string problem = problems[i];
        std::vector<std::thread> threads;

        uint64_t now =  time_since_epoch_milliseconds();
        std::cout << "Time taken: " << now - previous << ":" << problem << std::endl;
        previous = now;

        for (int i = 0; i < NUM_THREADS; i++) {
            threads.push_back(std::thread(run_benchmark, problem));
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            threads[i].join();
        }
    }


    std::cout << time_since_epoch_milliseconds() - previous << std::endl;

    std::cout << "End: " << time_since_epoch_milliseconds() << std::endl;
}