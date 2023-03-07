#ifndef FUNCTION
#define FUNCTION

#include "../data/data.h"
#include <iostream>
#include "../structures/structure.h"

#define FULL_MASK 0xffffffff


template <class A, class B>
class Function{
    bool _is_on_gpu = false;
    bool _is_on_scheduler = false;

    Function* copied_from = nullptr;

    // std::vector<Function*> sources;
    // int source_offset = 0;

    // std::vector<Function*> targets;
    // int target_offset = 0;
public:
    int initialization_method = -1;

    // Feed forward data
    Data<A> input; 
    Data<B> output; 

    // Backpropagation data
    Data<float> input_weights; 
    Data<float> output_weights;
    
    // Parameter weights of this function
    Data<float> weights;

    // Computed delta after backpropagation
    Data<float> delta_weights;
    
    bool forward_fed;

    __host__ __device__ Function();
    __host__ __device__ Function(int weight_count, int input_size, int output_size);
    __host__ __device__ Function(int weight_count, Function* source, int output_size, int source_offset=0);
    __host__ __device__ Function(int weight_count, int input_size, int output_size, Function* target, int target_offset=0, Data<float>* weights=nullptr);
    __host__ __device__ Function(int weight_count, int input_size, int output_size, Function* source, Function* target, int source_offset=0, int target_offset=0);
    __host__ __device__ Function(Function* source);

    void add_source(Function* source);
    void add_target(Function* target);

    bool is_on_gpu();

    virtual int object_size() {
        return sizeof(Function<A, B>);
    };

    // __device__ __host__ virtual Data<B> operator()(Data<A> input);
    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();

    virtual void move_to_gpu();
    virtual void move_to_scheduler(StructureScheduler* scheduler);

    __device__ __host__ virtual void apply_delta(float multiplier);
};

template <class A, class B>
void Function<A, B>::move_to_gpu() {
    // if (_is_on_gpu) return;
    
    // _is_on_gpu = true;

    // // Move all data to gpu
    // if (!input.is_on_gpu()) {
        
    //     input.move_to_gpu();
    // }
    
    // if (!output.is_on_gpu()) {
    //     output.move_to_gpu();
    // }
    
    // if (!input_weights.is_on_gpu()) {
    //     input_weights.move_to_gpu();
    // }
    
    // if (!output_weights.is_on_gpu()) {
    //     output_weights.move_to_gpu();
    // }
    
    // // Weights are never shared with any other function. Therefore we don't have to check
    // if (!weights.is_on_gpu()) weights.move_to_gpu();
    // if (!delta_weights.is_on_gpu()) delta_weights.move_to_gpu();
    
    // for (int i = 0; i < sources.size(); i++) {
    //     sources[i]->move_to_gpu();
    // }
    
    // for (int i = 0; i < targets.size(); i++) {
    //     targets[i]->move_to_gpu();
    // }
}

template <class A, class B>
void Function<A, B>::move_to_scheduler(StructureScheduler* scheduler) {}

template <class A, class B>
__host__ __device__ Function<A, B>::Function(){
    forward_fed = false;
    initialization_method = 1;
}

template <class A, class B>
__host__ __device__ Function<A, B>::Function(int weight_count, int input_size, int output_size){
    forward_fed = false;
    initialization_method = 2;

    weights.initialize(weight_count);
    delta_weights.initialize(weight_count);

    input.initialize(input_size);
    input_weights.initialize(input_size);

    output.initialize(output_size);
    output_weights.initialize(output_size);
}

template<class A, class B>
__host__ __device__ Function<A, B>::Function(int weight_count, Function* _source, int output_size, int source_offset) {
    forward_fed = false;
    initialization_method = 3;

    weights.initialize(weight_count);
    delta_weights.initialize(weight_count);

    input = _source->output;
    input_weights = _source->output_weights;

    output.initialize(output_size);
    output_weights.initialize(output_size);
}


template<class A, class B>
__host__ __device__ Function<A, B>::Function(int weight_count, int input_size, int output_size, Function* _target, int target_offset, Data<float>* _weights) :
weights(_weights, 0, weight_count)
{
    forward_fed = false;
    initialization_method = 4;

    // Weights
    delta_weights.initialize(weight_count);

    input.initialize(input_size);
    input_weights.initialize(input_size);

    #ifdef __CUDA_ARCH__
    if (_target->input.length() < target_offset + output_size) {
        printf("Target input of size %i is too small. Offset is %i and output size is %i. \n", input.length(), target_offset, output_size);
    }
    #endif

    output = _target->input.range(target_offset, target_offset + output_size);
    output_weights = _target->input_weights.range(target_offset, target_offset + output_size);

    weights[0] = 1.0;
}


template<class A, class B>
__host__ __device__ Function<A, B>::Function(int weight_count, int input_size, int output_size, Function* source, Function* target, int source_offset, int target_offset) {}


template<class A, class B>
__host__ __device__ Function<A, B>::Function(Function* copy_from) {
    initialization_method = 6;
    // throw;

    weights = copy_from->weights;
    delta_weights = copy_from->delta_weights;

    input = copy_from->input;
    output = copy_from->output;

    input_weights = copy_from->input_weights;
    output_weights = copy_from->output_weights;

    forward_fed = copy_from->forward_fed;

    copied_from = copy_from;
}

template <class A, class B>
void Function<A, B>::add_source(Function<A, B>* source) {}

template <class A, class B>
void Function<A, B>::add_target(Function<A, B>* target) {}

template <class A, class B>
bool Function<A, B>::is_on_gpu() {
    return _is_on_gpu;
}

template <class A, class B>
__device__ __host__ void Function<A, B>::feed_forward(){}


template <class A, class B>
__device__ __host__ void Function<A, B>::backpropagate(){}

template <class A, class B>
__device__ __host__ void Function<A, B>::apply_delta(float multiplier) {
#ifdef __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int chunk = 0; chunk < delta_weights.length(); chunk += 32) {
        if (chunk + warp_index < weights.length()) {
            float previous_weight = weights[chunk + warp_index];
            weights[chunk + warp_index] += delta_weights[chunk + warp_index] * multiplier;
            if (delta_weights.length() == 1056 && delta_weights[chunk + warp_index] != 0 && chunk + warp_index == 100) 
            printf("(%i/%i) %f + %f = %f, %p\n", chunk + warp_index, weights.length(), previous_weight, delta_weights[chunk + warp_index] * multiplier, weights[chunk + warp_index], &weights[0]);
        }
    }
#else
    for (int i = 0; i < delta_weights.length(); i++) {
        weights[i] += delta_weights[i] * multiplier;
    }
#endif
}

#endif