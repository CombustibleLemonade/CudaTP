#ifndef DENSE
#define DENSE

#include "function.h"

class Dense : public Function<float, float> {
    int bias_index;
    bool sigmoid_activation;
public:
    __host__ __device__ Dense(int input_size, int output_size, Data<float>* _weights=nullptr, bool sigmoid_activation=true);
    __host__ __device__ Dense(int input_size, int output_size, Function* target, int target_offset=0, Data<float>* _weights=nullptr, bool sigmoid_activation=true);
    __host__ __device__ Dense(Function* source, Function* target);
    __host__ __device__ Dense();
    
    virtual int object_size() {
        return sizeof(Dense);
    };
    
    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();
};

#endif