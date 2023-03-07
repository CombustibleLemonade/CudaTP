#include "copy.h"

__host__ __device__ Copy::Copy(int size, Function<float, float>* source, Function<float, float>* target, int source_offset, int target_offset) : 
    Function(0, size, size, source, target, source_offset, target_offset) {}

__device__ __host__ void Copy::feed_forward() {
#ifdef __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        if (chunk + warp_index < input.length()) {
            output[chunk + warp_index] = input[chunk + warp_index];
        }
    }
#endif
}

__device__ __host__ void Copy::backpropagate() {
#ifdef __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        if (chunk + warp_index < input.length()) {
            input_weights[chunk + warp_index] += output_weights[chunk + warp_index];
        }
    }
#endif
}