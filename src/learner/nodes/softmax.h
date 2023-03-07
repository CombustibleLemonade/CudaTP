#ifndef SOFTMAXX
#define SOFTMAXX

#include "function.h"

class Softmax : public Function<float, float> {
    // int _kernel;
    // int _width;
public:
    __host__ __device__ Softmax(int input_size);
    __host__ __device__ Softmax(Function* source);

    virtual int object_size() {
        return sizeof(Softmax);
    };

    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();
};

void call_softmax_cuda();

#endif