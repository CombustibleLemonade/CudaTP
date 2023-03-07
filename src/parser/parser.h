#ifndef PARSER
#define PARSER

#include "../prover/cudaProver.h"

void parse_file(const char* file_name, CudaTheorem* state);

CudaTheorem* parse_file(const char* file_name);

#endif