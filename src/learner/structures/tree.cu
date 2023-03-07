#include "tree.h"

#include <iostream>
#include <map>
#include "../nodes/softmax.h"
#include "../nodes/crossentropy.h"

#define THROUGHPUT_SIZE 32 // Size of layer data
#define MERGE_WEIGHTS_SIZE (2 * THROUGHPUT_SIZE) + 1 // Size of the merge layer
#define LITERAL_WEIGHTS_SIZE (2 * THROUGHPUT_SIZE) * THROUGHPUT_SIZE + THROUGHPUT_SIZE // Size of the literal layer
#define PAIR_WEIGHTS_SIZE (2 * THROUGHPUT_SIZE) * THROUGHPUT_SIZE + THROUGHPUT_SIZE // Size of the pair layer
#define VARIABLE_WEIGHTS_SIZE THROUGHPUT_SIZE * THROUGHPUT_SIZE + THROUGHPUT_SIZE // Size of the variable layer
#define CONSTANT_WEIGHTS_SIZE THROUGHPUT_SIZE * THROUGHPUT_SIZE + THROUGHPUT_SIZE // Size of the constant layer

// Objects have to be created on the GPU, otherwise virtual functions wont work
template <class T, typename... Arguments>
__global__ void cuda_new_global(T* t, Arguments... args) {
    new (t) T(args...);
}

template <class T, typename... Arguments>
T* cuda_new(Arguments... args) {
    T* result_address[1];
    cudaMalloc(result_address, sizeof(T*));
    T* result = result_address[0];
    cuda_new_global<<<1, 1>>>(result_address[0], args...);

    if (result == nullptr) {
        cudaDeviceSynchronize();
        std::cout << "Result is nullpointer" << std::endl;
        throw;
    }

    return result;
}

void register_dependency(StructureTaskWrapper* from, StructureTaskWrapper* to) {
    from->dependants.push_back(to);
    to->dependencies.push_back(from);
}

TheoremStructure::TheoremStructure(CudaTheorem* _theorem, ProbabilityMatrix* _matrix) : Structure::Structure() {
    theorem = _theorem;
    matrix = _matrix;
}

struct TheoremTaskStackFrame {
    CudaPair pair;
    int pair_index = 0;
    int node;

    StructureTaskWrapper* feedforward;
    StructureTaskWrapper* backpropagation;
};

template<class T, typename... Arguments> 
StructureTaskWrapper* create_task(Function<float, float>* layer, Arguments... args) {
    T* task = cuda_new<T>(layer, args...);
    StructureTaskWrapper* result = new StructureTaskWrapper(task, layer);

    return result;
}

void TheoremStructure::apply_weights_delta(std::vector<StructureTaskWrapper*>* tasks, std::vector<StructureTaskWrapper*>* backpropagation_tasks) {
    StructureTaskWrapper* previous_copy_task = nullptr;

    for (int i = 0; i < backpropagation_tasks->size(); i++){
        StructureTaskWrapper* source_task_wrapper = (*backpropagation_tasks)[i];
        StructureTaskWrapper* task_wrapper = create_task<ApplyWeightsDeltaTask>(source_task_wrapper->layer);
        task_wrapper->info = "weight application";

        for (int j = 0; j < backpropagation_tasks->size(); j++) {
            register_dependency((*backpropagation_tasks)[i], task_wrapper);
        }

        register_dependency(source_task_wrapper, task_wrapper);
        tasks->push_back(task_wrapper);

        if (previous_copy_task != nullptr) {
            register_dependency(previous_copy_task, task_wrapper);
        }

        previous_copy_task = task_wrapper;
    }
}

void TheoremStructure::add_tasks(std::vector<StructureTaskWrapper*>* tasks) {
    // Initialize weights
    merge_layer_weights = cuda_new<Data<float>>(MERGE_WEIGHTS_SIZE);
    literal_layer_weights = cuda_new<Data<float>>(LITERAL_WEIGHTS_SIZE);
    pair_layer_weights = cuda_new<Data<float>>(PAIR_WEIGHTS_SIZE);
    variable_layer_weights = cuda_new<Data<float>>(VARIABLE_WEIGHTS_SIZE);
    constant_layer_weights = cuda_new<Data<float>>(CONSTANT_WEIGHTS_SIZE);

    // List of backpropagation task with weights. Used for delta weight application mutex.
    std::vector<StructureTaskWrapper*> merge_backpropagation_tasks;
    std::vector<StructureTaskWrapper*> literal_backpropagation_tasks;
    std::vector<StructureTaskWrapper*> pair_backpropagation_tasks;
    std::vector<StructureTaskWrapper*> variable_backpropagation_tasks;
    std::vector<StructureTaskWrapper*> constant_backpropagation_tasks;
    
    std::vector<Function<float, float>*> layers;

    std::map<std::pair<int, int>, StructureTaskWrapper*> literal_feedforward_task_wrappers;
    std::map<std::pair<int, int>, StructureTaskWrapper*> literal_backpropagation_task_wrappers;

    std::map<std::pair<int, int>, StructureTaskWrapper*> previous_copy_backpropagation_task_wrappers;

    for (int c = 0; c < theorem->clause_list.length; c++) {
        for (int l = 0; l < theorem->clause_list.clauses[c].length; l++) {
            if ( (matrix->literals.find(std::make_pair(c, l)) == matrix->literals.end()) ) continue;
            
            // Set the previous copy task wrapper to nullptr for each clause and literal
            previous_copy_backpropagation_task_wrappers[std::make_pair(c, l)] = nullptr;

            // Retrieve literal
            CudaLiteral literal = theorem->clause_list.clauses[c].literals[l];

            // Create layer
            Dense* literal_layer = cuda_new<Dense>(32, 1, literal_layer_weights);            
            layers.push_back(literal_layer);

            // Wrappers for feed forward
            StructureTaskWrapper* literal_feedforward_wrapper = create_task<LiteralFeedforwardTask>(literal_layer); 
            tasks->push_back(literal_feedforward_wrapper);

            // Add it to the dictionary
            literal_feedforward_task_wrappers[std::make_pair(c, l)] = literal_feedforward_wrapper;

            // Wrappers for backpropagation
            StructureTaskWrapper* literal_backpropagation_wrapper = create_task<LiteralBackpropagationTask>(literal_layer);

            // Add to tasks
            tasks->push_back(literal_backpropagation_wrapper);
            literal_backpropagation_tasks.push_back(literal_backpropagation_wrapper);

            // Add it to the dictionary
            literal_backpropagation_task_wrappers[std::make_pair(c, l)] = literal_backpropagation_wrapper;

            // Initialize stack for loop
            std::vector<TheoremTaskStackFrame> stack;
            
            stack.emplace_back();
            stack.back().node = literal.node;
            stack.back().pair = theorem->pair_list.pair_list[literal.node & ~NODE_TYPE_MASK];
            stack.back().feedforward = literal_feedforward_wrapper;
            stack.back().backpropagation = literal_backpropagation_wrapper;
            
            while (stack.size() > 0) {
                if (stack.back().pair_index == 2 || stack.size() == 1 && stack.back().pair_index == 1) {
                    stack.pop_back();
                }
                else {
                    stack.back().pair_index++;

                    switch (stack.back().node & NODE_TYPE_MASK) {
                    case PAIR_MASK:
                    {
                        // Layer
                        Dense* pair_layer = cuda_new<Dense>(64, 32, stack.back().feedforward->layer, (stack.back().pair_index - 1) * 32, pair_layer_weights); 
                        layers.push_back(pair_layer);

                        // Feed-forward task
                        StructureTaskWrapper* pair_feedforward_wrapper = create_task<PairFeedforwardTask>(pair_layer);
                        pair_feedforward_wrapper->info = "Pair feed-forward for input " + std::to_string(stack.back().pair_index);
                        register_dependency(pair_feedforward_wrapper, stack.back().feedforward);

                        // Backpropagation task
                        StructureTaskWrapper* pair_backpropagation_wrapper = create_task<PairBackpropagationTask>(pair_layer);
                        register_dependency(stack.back().backpropagation, pair_backpropagation_wrapper);

                        // Add to tasks
                        tasks->push_back(pair_feedforward_wrapper);
                        tasks->push_back(pair_backpropagation_wrapper);
                        pair_backpropagation_tasks.push_back(pair_backpropagation_wrapper);

                        // Add to the pair index;
                        CudaPair pair = stack.back().pair;
                        
                        // Add to the stack
                        stack.emplace_back();
                        stack.back().node = stack[stack.size() - 2].pair_index == 1 ? pair.a : pair.b;
                        stack.back().pair = theorem->pair_list.pair_list[stack.back().node & ~NODE_TYPE_MASK];
                        stack.back().feedforward = pair_feedforward_wrapper;
                        stack.back().backpropagation = pair_backpropagation_wrapper;
                    }
                    break;
                    case VARIABLE_MASK:
                    {
                        Dense* variable_layer = cuda_new<Dense>(32, 32, stack.back().feedforward->layer, (stack.back().pair_index - 1) * 32, variable_layer_weights);
                        layers.push_back(variable_layer);

                        // Feed-forward task 
                        StructureTaskWrapper* variable_feedforward_wrapper = create_task<VariableFeedforwardTask>(variable_layer, TextInput32(Names::names[stack.back().node].c_str()));
                        variable_feedforward_wrapper->info = "Variable feed-forward for input " + std::to_string(stack.back().pair_index);
                        register_dependency(variable_feedforward_wrapper, stack.back().feedforward);

                        // Add to tasks
                        tasks->push_back(variable_feedforward_wrapper);

                        // Backpropagation task
                        StructureTaskWrapper* variable_backpropagation_task_wrapper = create_task<VariableBackpropagationTask>(variable_layer, TextInput32(Names::names[stack.back().node].c_str()));
                        register_dependency(stack.back().backpropagation, variable_backpropagation_task_wrapper);

                        // Add to tasks
                        tasks->push_back(variable_backpropagation_task_wrapper);
                        variable_backpropagation_tasks.push_back(variable_backpropagation_task_wrapper);
                    }
                    break;
                    case CONSTANT_MASK:
                    {
                        Dense* constant_layer = cuda_new<Dense>(32, 32, stack.back().feedforward->layer, (stack.back().pair_index - 1) * 32, constant_layer_weights);
                        layers.push_back(constant_layer);

                        // Feed-forward task
                        StructureTaskWrapper* constant_feedforward_task_wrapper = create_task<ConstantFeedforwardTask>(constant_layer, TextInput32(Names::names[stack.back().node].c_str()));
                        constant_feedforward_task_wrapper->info = "Constant feed-forward for input " + std::to_string(stack.back().pair_index);
                        register_dependency(constant_feedforward_task_wrapper, stack.back().feedforward);

                        // Add to tasks
                        tasks->push_back(constant_feedforward_task_wrapper);

                        // Backpropagation task
                        StructureTaskWrapper* constant_backpropagation_task_wrapper = create_task<ConstantBackpropagationTask>(constant_layer, TextInput32(Names::names[stack.back().node].c_str())); 
                        register_dependency(stack.back().backpropagation, constant_backpropagation_task_wrapper);

                        // Add to tasks
                        tasks->push_back(constant_backpropagation_task_wrapper);
                        
                        if (constant_backpropagation_task_wrapper->task == nullptr) std::cout << "NULL TASK" << std::endl;

                        constant_backpropagation_tasks.push_back(constant_backpropagation_task_wrapper);
                    }
                    break;
                    }
                }
            }
        }
    }

    output_length = matrix->literals.size();
    cudaMalloc(&network_output, sizeof(float) * output_length);
    // std::cout << "asiudyfgiuy" << matrix->literals.size() << std::endl;

    // Add all possible pairs using unification matrix
    std::cout << "Adding all possible pairs" << std::endl;
    int output_index = 0;

    for (int c = 0; c < theorem->clause_list.length; c++) {
        for (int l = 0; l < theorem->clause_list.clauses[c].length; l++) {
            if (matrix->literals.find(std::make_pair(c, l)) == matrix->literals.end()) continue;
            CudaMatrixIndex index = theorem->matrix.indices[c*8 + l];

            // Create softmax task
            Softmax* softmax = cuda_new<Softmax>(index.length);

            StructureTaskWrapper* softmax_feedforward_task_wrapper = create_task<SoftmaxFeedforwardTask>(softmax);
            softmax_feedforward_task_wrapper->info = "Softmax feed-forward";
            tasks->push_back(softmax_feedforward_task_wrapper);

            StructureTaskWrapper* softmax_backpropagation_task_wrapper = create_task<SoftmaxBackpropagationTask>(softmax);
            softmax_backpropagation_task_wrapper->info = "Softmax backpropagation";
            tasks->push_back(softmax_backpropagation_task_wrapper);
            
            // Create crossentropy task
            float* distribution = new float[index.length];
            for (int i = 0; i < index.length; i++) {
                int to = theorem->matrix.list[index.index + i];
                int c_to = to / 8;
                int l_to = to % 8;
                distribution[i] = matrix->weights[{c, l, c_to, l_to}];
            }

            float* gpu_distribution;
            cudaMalloc(&gpu_distribution, index.length * sizeof(float));
            cudaMemcpy(gpu_distribution, distribution, index.length * sizeof(float), cudaMemcpyHostToDevice);

            Crossentropy* crossentropy = cuda_new<Crossentropy>(softmax, gpu_distribution);

            StructureTaskWrapper* crossentropy_feedforward_task_wrapper = create_task<CrossentropyFeedforwardTask>(crossentropy, network_output, output_index++);
            crossentropy_feedforward_task_wrapper->info = "Crossentropy feed-forward";
            register_dependency(softmax_feedforward_task_wrapper, crossentropy_feedforward_task_wrapper);
            tasks->push_back(crossentropy_feedforward_task_wrapper);
            
            StructureTaskWrapper* crossentropy_backpropagation_task_wrapper = create_task<CrossentropyBackpropagationTask>(crossentropy);
            crossentropy_backpropagation_task_wrapper->info = "Crossentropy backpropagation";
            register_dependency(crossentropy_feedforward_task_wrapper, crossentropy_backpropagation_task_wrapper);
            register_dependency(crossentropy_backpropagation_task_wrapper, softmax_backpropagation_task_wrapper);
            tasks->push_back(crossentropy_backpropagation_task_wrapper);

            // Create copy tasks that copy data to and from the softmax crossentropy tasks
            for (int i = 0; i < index.length; i++) {
                int to = theorem->matrix.list[index.index + i];
                int c_to = to / 8;
                int l_to = to % 8;

                // Create the merge task
                Dense* merge = cuda_new<Dense>(64, 1, softmax, i, merge_layer_weights, false);
                
                StructureTaskWrapper* merge_feedforward_task_wrapper = create_task<MergeFeedforwardTask>(merge);
                merge_feedforward_task_wrapper->info = "Left merge feed-forward";
                register_dependency(merge_feedforward_task_wrapper, softmax_feedforward_task_wrapper);
                tasks->push_back(merge_feedforward_task_wrapper);

                StructureTaskWrapper* merge_backpropagation_task_wrapper = create_task<MergeBackpropagationTask>(merge);
                merge_backpropagation_task_wrapper->info = "Left merge backpropagation";
                register_dependency(softmax_backpropagation_task_wrapper, merge_backpropagation_task_wrapper);
                
                tasks->push_back(merge_backpropagation_task_wrapper);
                merge_backpropagation_tasks.push_back(merge_backpropagation_task_wrapper);

                // Copy the left input of the merge layer
                {
                    StructureTaskWrapper* literal_feedforward_task_wrapper = literal_feedforward_task_wrappers[std::make_pair(c, l)];
                    StructureTaskWrapper* literal_backpropagation_task_wrapper = literal_backpropagation_task_wrappers[std::make_pair(c, l)];

                    // Copy to the first input
                    Copy* copy_left = cuda_new<Copy>(32, literal_feedforward_task_wrapper->layer, merge, 0, 0);

                    // Feedforward task
                    StructureTaskWrapper* copy_feedforward_task_wrapper = create_task<CopyFeedforwardTask>(copy_left);
                    register_dependency(literal_feedforward_task_wrapper, copy_feedforward_task_wrapper);
                    register_dependency(copy_feedforward_task_wrapper, merge_feedforward_task_wrapper);

                    tasks->push_back(copy_feedforward_task_wrapper);

                    // Backpropagation task
                    StructureTaskWrapper* copy_backpropagation_task_wrapper = create_task<CopyBackpropagationTask>(copy_left);

                    StructureTaskWrapper* previous_copy_task = previous_copy_backpropagation_task_wrappers[std::make_pair(c, l)];
                    if (previous_copy_task != nullptr) {
                        register_dependency(previous_copy_task, copy_backpropagation_task_wrapper);
                    } else {
                        register_dependency(merge_backpropagation_task_wrapper, copy_backpropagation_task_wrapper);
                    }
                    register_dependency(copy_backpropagation_task_wrapper, literal_backpropagation_task_wrapper);
                    previous_copy_backpropagation_task_wrappers[std::make_pair(c, l)] = copy_backpropagation_task_wrapper;

                    tasks->push_back(copy_backpropagation_task_wrapper);
                }

                // Copy the right input of the merge layer
                {
                    StructureTaskWrapper* literal_feedforward_task_wrapper = literal_feedforward_task_wrappers[std::make_pair(c_to, l_to)];
                    StructureTaskWrapper* literal_backpropagation_task_wrapper = literal_backpropagation_task_wrappers[std::make_pair(c_to, l_to)];

                    // Copy to the second input
                    Copy* copy_right = cuda_new<Copy>(32, literal_feedforward_task_wrapper->layer, merge, 0, 32);

                    // Feedforward task
                    CopyFeedforwardTask* copy_feedforward_task = cuda_new<CopyFeedforwardTask>(copy_right);
                    StructureTaskWrapper* copy_feedforward_task_wrapper = new StructureTaskWrapper(copy_feedforward_task, copy_right);
                    register_dependency(literal_feedforward_task_wrapper, copy_feedforward_task_wrapper);
                    register_dependency(copy_feedforward_task_wrapper, merge_feedforward_task_wrapper);

                    tasks->push_back(copy_feedforward_task_wrapper);

                    // Backpropagation task
                    CopyBackpropagationTask* copy_backpropagation_task = cuda_new<CopyBackpropagationTask>(copy_right);
                    StructureTaskWrapper* copy_backpropagation_task_wrapper = new StructureTaskWrapper(copy_backpropagation_task, copy_right);

                    StructureTaskWrapper* previous_copy_task = previous_copy_backpropagation_task_wrappers[std::make_pair(c_to, l_to)];
                    if (previous_copy_task != nullptr) {
                        register_dependency(previous_copy_task, copy_backpropagation_task_wrapper);
                    } else {
                        register_dependency(merge_backpropagation_task_wrapper, copy_backpropagation_task_wrapper);
                    }
                    register_dependency(copy_backpropagation_task_wrapper, literal_backpropagation_task_wrapper);
                    previous_copy_backpropagation_task_wrappers[std::make_pair(c, l)] = copy_backpropagation_task_wrapper;

                    tasks->push_back(copy_backpropagation_task_wrapper);
                }
            }
        }
    }

    std::cout << "Adding weight delta tasks" << std::endl;
    apply_weights_delta(tasks, &merge_backpropagation_tasks);
    apply_weights_delta(tasks, &literal_backpropagation_tasks);
    apply_weights_delta(tasks, &pair_backpropagation_tasks);
    apply_weights_delta(tasks, &variable_backpropagation_tasks);
    apply_weights_delta(tasks, &constant_backpropagation_tasks);

    std::cout << "Computing longest paths" << std::endl;
    // Compute longest paths
    for (int i = 0; i < tasks->size(); i++){
        StructureTaskWrapper* task_wrapper = (*tasks)[i];
        if (task_wrapper->dependants.size() == 0) {
            task_wrapper->distance_to_end = 0;
        }
    }

    int distance = 0;
    while (true) {
        int hits = 0;
        for (int i = 0; i < tasks->size(); i++) {
            StructureTaskWrapper* task_wrapper = (*tasks)[i];
            if (task_wrapper->distance_to_end == distance) {
                for (int j = 0; j < task_wrapper->dependencies.size(); j++) {
                    task_wrapper->dependencies[j]->distance_to_end = distance + 1;
                    hits++;
                }
            }
        }

        // std::cout << distance << "\t" << hits << std::endl;

        if (hits == 0) break;
        distance++;
    }
}

__global__ void print_weights(Data<float>* data) {
    for (int i = 0; i < data->length(); i++) {
        printf("%f, %p\n", (*data)[i], &(*data)[i]);
    }
}

void TheoremStructure::print_variable_weights() {
    print_weights<<<1, 1>>>(merge_layer_weights);
    cudaDeviceSynchronize();
}

__host__ __device__ void LiteralFeedforwardTask::perform_task() {
    layer->feed_forward();
}

__host__ __device__ LiteralFeedforwardTask::LiteralFeedforwardTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void LiteralBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ LiteralBackpropagationTask::LiteralBackpropagationTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void PairFeedforwardTask::perform_task() {
    layer->feed_forward();
}

__host__ __device__ PairFeedforwardTask::PairFeedforwardTask() {}

__host__ __device__ PairFeedforwardTask::PairFeedforwardTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void PairBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ PairBackpropagationTask::PairBackpropagationTask() {}

__host__ __device__ PairBackpropagationTask::PairBackpropagationTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void VariableFeedforwardTask::perform_task() {
    layer->feed_forward();
}

__host__ __device__ VariableFeedforwardTask::VariableFeedforwardTask(Function<float, float>* _layer, const char* _source) : StructureTask(_layer) {}

__host__ __device__ VariableFeedforwardTask::VariableFeedforwardTask(Function<float, float>* _layer, TextInput32 _source) : StructureTask(_layer) {
    _source.embed(&_layer->input[0]);
    // printf("Variable: %c%c%c%c\n", _source.text[0], _source.text[1], _source.text[2], _source.text[3]);
}


__host__ __device__ void VariableBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ VariableBackpropagationTask::VariableBackpropagationTask(Function<float, float>* _layer, const char* _source) : StructureTask(_layer) {}

__host__ __device__ VariableBackpropagationTask::VariableBackpropagationTask(Function<float, float>* _layer, TextInput32 _source) : StructureTask(_layer) {}


__host__ __device__ void ConstantFeedforwardTask::perform_task() {
    layer->feed_forward();
} 

__host__ __device__ ConstantFeedforwardTask::ConstantFeedforwardTask(Function<float, float>* _layer, const char* source) : StructureTask(_layer) {}

__host__ __device__ ConstantFeedforwardTask::ConstantFeedforwardTask(Function<float, float>* _layer, TextInput32 _source) : StructureTask(_layer) {
    _source.embed(&_layer->input[0]);
    // printf("Constant: %c%c%c%c\n", _source.text[0], _source.text[1], _source.text[2], _source.text[3]);

}


__host__ __device__ void ConstantBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ ConstantBackpropagationTask::ConstantBackpropagationTask(Function<float, float>* _layer, const char* _source) : StructureTask(_layer) {}

__host__ __device__ ConstantBackpropagationTask::ConstantBackpropagationTask(Function<float, float>* _layer, TextInput32 _source) : StructureTask(_layer) {}


__host__ __device__ void MergeFeedforwardTask::perform_task() {
    layer->feed_forward();
}

__host__ __device__ MergeFeedforwardTask::MergeFeedforwardTask() {}

__host__ __device__ MergeFeedforwardTask::MergeFeedforwardTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void MergeBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ MergeBackpropagationTask::MergeBackpropagationTask() {}

__host__ __device__ MergeBackpropagationTask::MergeBackpropagationTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void SoftmaxFeedforwardTask::perform_task() {
    layer->feed_forward();
}

__host__ __device__ SoftmaxFeedforwardTask::SoftmaxFeedforwardTask() {}

__host__ __device__ SoftmaxFeedforwardTask::SoftmaxFeedforwardTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void SoftmaxBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ SoftmaxBackpropagationTask::SoftmaxBackpropagationTask() {}

__host__ __device__ SoftmaxBackpropagationTask::SoftmaxBackpropagationTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void CrossentropyFeedforwardTask::perform_task() {
    layer->feed_forward();

    crossentropy_outputs[output_index] = layer->output[0];
}

__host__ __device__ CrossentropyFeedforwardTask::CrossentropyFeedforwardTask() {}

__host__ __device__ CrossentropyFeedforwardTask::CrossentropyFeedforwardTask(Function<float, float>* _layer, float* _crossentropy_outputs, int index) : StructureTask(_layer) {
    crossentropy_outputs = _crossentropy_outputs;
    output_index = index;
}


__host__ __device__ void CrossentropyBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ CrossentropyBackpropagationTask::CrossentropyBackpropagationTask() {}

__host__ __device__ CrossentropyBackpropagationTask::CrossentropyBackpropagationTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void CopyFeedforwardTask::perform_task() {
    layer->feed_forward();
}

__host__ __device__ CopyFeedforwardTask::CopyFeedforwardTask() {}

__host__ __device__ CopyFeedforwardTask::CopyFeedforwardTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void CopyBackpropagationTask::perform_task() {
    layer->backpropagate();
}

__host__ __device__ CopyBackpropagationTask::CopyBackpropagationTask() {}

__host__ __device__ CopyBackpropagationTask::CopyBackpropagationTask(Function<float, float>* _layer) : StructureTask(_layer) {}


__host__ __device__ void ApplyWeightsDeltaTask::perform_task() {
    layer->apply_delta(100000.0);
}

__host__ __device__ ApplyWeightsDeltaTask::ApplyWeightsDeltaTask(Function<float, float>* _layer) : StructureTask(_layer) {}
