#include<iostream>
#include <thread>

#include "benchmark.h"
#include "parser/parser.h"
#include "prover/prover.h"

void run_benchmark(std::string path) {
    // std::cout << path << std::endl;
    CudaTheorem* theorem = parse_file(("data/problems/" + path + ".th").c_str());
    load_matrix_from_file(theorem, "data/weights/" + path + ".mtrx");

    CudaProofState state;
    state.theorem = theorem;
    
    start_prover(theorem);
}

void run_benchmark_thread(int index, int gpu) {
    cudaSetDevice(gpu);

    uint64_t previous = time_since_epoch_milliseconds();

    for (int i = 0; i + index < problems.size(); i++) {
        std::string problem = problems[i + index];

        uint64_t now =  time_since_epoch_milliseconds();
        std::cout << "Time taken: " << now - previous << ":" << problem << std::endl;
        previous = now;

        run_benchmark(problem);
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Benchmarking GPU" << std::endl;
    uint64_t start = time_since_epoch_milliseconds();
    
    std::vector<std::thread> threads;
    threads.push_back(std::thread(run_benchmark_thread, 0, 0));
    // threads.push_back(std::thread(run_benchmark_thread, 1, 2));
    // threads.push_back(std::thread(run_benchmark_thread, 2, 3));

    // run_benchmark_thread(0);

    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    std::cout << time_since_epoch_milliseconds() - start << std::endl;
}