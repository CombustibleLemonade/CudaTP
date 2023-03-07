#ifndef CUDA_PROVER
#define CUDA_PROVER

#include <stdint.h>
#include <string>

#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define CLAUSE_WIDTH 8
#define CLAUSE_COUNT 2048

#define NODE_TYPE_MASK 0xC0000000

#define PAIR_MASK       0x40000000
#define VARIABLE_MASK   0x80000000
#define CONSTANT_MASK   0xC0000000

#define CUDA_MAX_ACTIONS 512
#define CUDA_MAX_STACK 128

#define CUDA_SUBSTITUTION_TABLE_SIZE 4096

#define CUDA_PAIR_LIST_LENGTH 8192

#define MAX_OCCURS_STACK_SIZE 256


#define BLOCK_COUNT 256
#define THREAD_PER_BLOCK 128

enum CudaActionType{
    EXTEND,
    REDUCE,
    RETRACT
};

class CudaPair{
public:
    int a;
    int b;
};

class CudaPairList{
public:
    int length = 0;
    CudaPair pair_list[CUDA_PAIR_LIST_LENGTH];
};

class CudaLiteral{
public:
    int clause_idx = -1;
    int literal_idx = -1;
    int node = -1;
    bool negative = false;
};

class CudaClause{
public:
    int length = 0;
    CudaLiteral literals[CLAUSE_WIDTH];
};

class CudaClauseList{
public:
    int length = 0;
    CudaClause clauses[CLAUSE_COUNT];
};

class CudaMatrixIndex{
public:
    int index = -1;
    int length = -1;
};

class CudaMatrix{
public:
    int list[CLAUSE_COUNT * CLAUSE_COUNT * 8];
    float probabilities[CLAUSE_COUNT * CLAUSE_COUNT * 8];
    CudaMatrixIndex indices[CLAUSE_COUNT * CLAUSE_WIDTH];
};

class CudaTheorem{
public:
    CudaPairList pair_list;
    CudaClauseList clause_list;
    CudaMatrix matrix;

    __host__ __device__ bool is_valid_literal(CudaLiteral* l);
    __host__ bool fill_matrix();
};

class CudaSubstitution{
public:
    int from_node = -1;
    int from_sub_idx = -1;

    int to_node = -1;
    int to_sub_idx = -1;

    CudaSubstitution* previous = nullptr;

    __host__ __device__ bool occupied();
    __host__ __device__ void erase();
};

class CudaSubstitutionTable{
public:
    CudaSubstitution substitutions[CUDA_SUBSTITUTION_TABLE_SIZE];

    int element_count = 0;
    CudaSubstitution* head = nullptr;

    __host__ __device__ CudaSubstitutionTable();

    __host__ __device__ CudaSubstitution* append(int node, int substitution_idx);
    __host__ __device__ CudaSubstitution* get(int node, int substitution_idx);
    __host__ __device__ void push(int substitution_idx);
    __host__ __device__ void pop();
    __host__ __device__ void decapitate();
};

class CudaActionLogEntry{
public:
    CudaLiteral* source = nullptr;
    CudaLiteral* target = nullptr;
    int substitution_index = -1;
    uint8_t visit_log = 0;
    CudaActionType type;
};

class CudaActionLog{
public:
    int action_length = 0;
    CudaActionLogEntry actions[CUDA_MAX_ACTIONS];

    int stack_length = 0;
    CudaActionLogEntry stack[CUDA_MAX_STACK];

    bool ready = true;

    float likelihood = 1.0;

    __host__ __device__ void undo();
    __host__ __device__ void register_action(CudaActionLogEntry action);
    __host__ __device__ void backtrack();
};

class CudaMutation{
public:
    CudaSubstitutionTable table;
    CudaActionLog log;

    uint8_t visited = 0;
    int substitution_index = 1;
    CudaLiteral* head;

    bool solved = false;
    __host__ __device__ void tread_back();
};

class SolutionStep{
public:
    int from_clause_idx;
    int from_literal_idx;

    int to_clause_idx;
    int to_literal_idx;

    CudaActionType step_type;
};

class Solution{
public:
    int length;
    SolutionStep steps[CUDA_MAX_ACTIONS];
};

class CudaProofState{
    CudaSubstitution occurs_stack[MAX_OCCURS_STACK_SIZE];
    CudaSubstitution unify_stack[MAX_OCCURS_STACK_SIZE];


    int occurs_overflows = 0;

    __host__ __device__ int get_random_int(int min, int max);
public:
    CudaTheorem* theorem;
    CudaMutation mutation;

    curandState* random;

    int total_inferences = 0;
    int solve_count = 0;

    __host__ __device__ int pick_next_literal();

    __host__ __device__ bool occurs(CudaSubstitution sub);
    __host__ __device__ bool unify(int a, int a_idx, int b, int b_idx);
    __host__ __device__ bool unify(CudaLiteral* a, int idx_a, CudaLiteral* b, int idx_b);
    __host__ __device__ void register_dead_clauses();
    __host__ __device__ bool extend(CudaLiteral* l);
    __host__ __device__ bool reduce(int idx);
    __host__ __device__ void undo();
    __host__ __device__ CudaLiteral* get_random_literal();
    __host__ __device__ bool prove();
    __device__ void hash_test();
    __host__ std::string to_string(int node, int substitution_idx, int depth = 0);
};

class ComputationResult{
public:
    Solution solutions[BLOCK_COUNT * THREAD_PER_BLOCK];
    int solution_counts[BLOCK_COUNT * THREAD_PER_BLOCK];
    int total_solutions;
};

class ProofConfig{
public:
    int runs = 1000000;
    int depth = 32;
};

ComputationResult* start_prover(CudaTheorem* t);

#endif