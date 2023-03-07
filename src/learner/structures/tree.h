#ifndef TREE
#define TREE

#include <string>

#include "../nodes/dense.h"
#include "../nodes/copy.h"
#include "structure.h"
#include "../../prover/cudaProver.h"
#include "../../parser/matrix_parser.h"
#include "../../parser/names.h"

class CrossentropyFeedforwardTask;

class TheoremStructure : public Structure {
    CudaTheorem* theorem;
    ProbabilityMatrix* matrix;

    Data<float>* merge_layer_weights;
    Data<float>* literal_layer_weights;
    Data<float>* pair_layer_weights;
    Data<float>* variable_layer_weights;
    Data<float>* constant_layer_weights;

    std::vector<CrossentropyFeedforwardTask> crossentropy_feedforward_tasks;

    void apply_weights_delta(std::vector<StructureTaskWrapper*>* tasks, std::vector<StructureTaskWrapper*>* backpropagation_tasks);
public:
    TheoremStructure(CudaTheorem* theorem, ProbabilityMatrix* matrix);
    virtual void add_tasks(std::vector<StructureTaskWrapper*>* tasks);

    void print_variable_weights();
};

class LiteralFeedforwardTask : public StructureTask {
    int clause_index;
    int literal_index;

public:
    __host__ __device__ LiteralFeedforwardTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task();
};

class LiteralBackpropagationTask : public StructureTask {
    int clause_index;
    int literal_index;

public:
    __host__ __device__ LiteralBackpropagationTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task();
};

// Pair feed forward task
class PairFeedforwardTask : public StructureTask {
    int pair_index;
public:
    virtual int size() {
        return sizeof(PairFeedforwardTask);
    }

    __host__ __device__ PairFeedforwardTask();
    __host__ __device__ PairFeedforwardTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task();
};

// Pair backpropagation task
class PairBackpropagationTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(PairBackpropagationTask);
    }
    
    __host__ __device__ PairBackpropagationTask();
    __host__ __device__ PairBackpropagationTask(Function<float, float>* layer);
    
    __host__ __device__ virtual void perform_task();
};

// Variable feed forward task
class VariableFeedforwardTask : public StructureTask {
    int variable_index;
public:
    virtual int size() {
        return sizeof(VariableFeedforwardTask);
    }

    char source[32];

    __host__ __device__ VariableFeedforwardTask(Function<float, float>* layer, const char* source);
    __host__ __device__ VariableFeedforwardTask(Function<float, float>* layer, TextInput32 source);

    __host__ __device__ virtual void perform_task();
};

// Variable backpropagation task
class VariableBackpropagationTask : public StructureTask {
    int variable_index;
public:
    virtual int size() {
        return sizeof(VariableBackpropagationTask);
    }

    char source[32] = "wolla";
    
    __host__ __device__ VariableBackpropagationTask(Function<float, float>* layer, const char* _source);
    __host__ __device__ VariableBackpropagationTask(Function<float, float>* layer, TextInput32 source);

    __host__ __device__ virtual void perform_task();
};

// Constant feed forward task
class ConstantFeedforwardTask : public StructureTask {
    int constant_index;
public:
    virtual int size() {
        return sizeof(ConstantFeedforwardTask);
    }

    char source[32] = "wolla";
    
    __host__ __device__ ConstantFeedforwardTask(Function<float, float>* layer, const char* source);
    __host__ __device__ ConstantFeedforwardTask(Function<float, float>* layer, TextInput32 source);

    __host__ __device__ virtual void perform_task(); 
};

// Constant backpropagation task
class ConstantBackpropagationTask : public StructureTask {
    int constant_index;
public:
    virtual int size() {
        return sizeof(ConstantBackpropagationTask);
    }

    char source[32];

    __host__ __device__ ConstantBackpropagationTask(Function<float, float>* layer, const char* source);
    __host__ __device__ ConstantBackpropagationTask(Function<float, float>* layer, TextInput32 source);

    __host__ __device__ virtual void perform_task(); 
};


// Merge feed forward task
class MergeFeedforwardTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(MergeFeedforwardTask);
    }

    __host__ __device__ MergeFeedforwardTask();
    __host__ __device__ MergeFeedforwardTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Merge backpropagation task
class MergeBackpropagationTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(MergeBackpropagationTask);
    }

    __host__ __device__ MergeBackpropagationTask();
    __host__ __device__ MergeBackpropagationTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Softmax feed forward task
class SoftmaxFeedforwardTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(SoftmaxFeedforwardTask);
    }

    __host__ __device__ SoftmaxFeedforwardTask();
    __host__ __device__ SoftmaxFeedforwardTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Softmax backpropagation task
class SoftmaxBackpropagationTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(SoftmaxBackpropagationTask);
    }

    __host__ __device__ SoftmaxBackpropagationTask();
    __host__ __device__ SoftmaxBackpropagationTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Crossentropy feed forward task
class CrossentropyFeedforwardTask : public StructureTask {
    int output_index;
    float* crossentropy_outputs;
public:
    virtual int size() {
        return sizeof(CrossentropyFeedforwardTask);
    }

    float* target_space;

    __host__ __device__ CrossentropyFeedforwardTask();
    __host__ __device__ CrossentropyFeedforwardTask(Function<float, float>* layer, float* crossentropy_outputs, int index);

    __host__ __device__ virtual void perform_task(); 
};

// Crossentropy backpropagation task
class CrossentropyBackpropagationTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(CrossentropyBackpropagationTask);
    }

    __host__ __device__ CrossentropyBackpropagationTask();
    __host__ __device__ CrossentropyBackpropagationTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Copy feed forward task
class CopyFeedforwardTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(CopyFeedforwardTask);
    }

    __host__ __device__ CopyFeedforwardTask();
    __host__ __device__ CopyFeedforwardTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Copy backpropagation task
class CopyBackpropagationTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(CopyBackpropagationTask);
    }

    __host__ __device__ CopyBackpropagationTask();
    __host__ __device__ CopyBackpropagationTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

// Apply weights delta task, for after 
class ApplyWeightsDeltaTask : public StructureTask {
public:
    virtual int size() {
        return sizeof(ApplyWeightsDeltaTask);
    }

    __host__ __device__ ApplyWeightsDeltaTask(Function<float, float>* layer);

    __host__ __device__ virtual void perform_task(); 
};

class TheoremTree : Structure {
public:
    TheoremTree(std::vector<CudaTheorem*> t);
};

#endif