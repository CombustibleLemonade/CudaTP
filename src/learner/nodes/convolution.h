#ifndef CONVOLUTION
#define CONVOLUTION

#include "function.h"

class Convolution : public Function<float, float> {
    int _kernel;
    int _width;
public:
    __host__ __device__ Convolution(int kernel, int width);

    virtual int object_size() {
        return sizeof(Convolution);
    };

    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();
};

void call_cuda();

#endif