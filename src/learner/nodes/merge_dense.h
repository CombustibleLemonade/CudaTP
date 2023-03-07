#ifndef MERGE_DENSE
#define MERGE_DENSE

#include "function.h"

class MergeDense : public Function<float, float> {
public:
    __host__ __device__ MergeDense(Function* source, int output_size);
    

    virtual int object_size() {
        return sizeof(MergeDense);
    };

    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();
};

#endif