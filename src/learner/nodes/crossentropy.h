#ifndef CROSSENTROPY
#define CROSSENTROPY

#include "function.h"

class Crossentropy : public Function<float, float> {
    int distribution_size;
public:
    float* distribution;
    
    __host__ __device__ Crossentropy(int size, float* distribution);
    __host__ __device__ Crossentropy(Function* source, float* distribution);

    virtual int object_size() {
        return sizeof(Crossentropy);
    };

    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();

    virtual void move_to_gpu();
};

#endif