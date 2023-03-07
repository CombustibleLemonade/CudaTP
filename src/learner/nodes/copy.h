#ifndef COPY
#define COPY

#include "function.h"

class Copy : public Function<float, float> {
public:
    __host__ __device__ Copy(int size, Function<float, float>* source, Function<float, float>* target, int source_offset, int target_offset);

    virtual int object_size() {
        return sizeof(Copy);
    };

    __device__ __host__ virtual void feed_forward();
    __device__ __host__ virtual void backpropagate();
};

#endif