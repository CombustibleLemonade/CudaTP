#include "merge_dense.h"

__host__ __device__ MergeDense::MergeDense(Function* source, int output_size) : Function(source->output.length() * output_size, source, output_size) {}

__device__ __host__ void MergeDense::feed_forward() {
#ifdef  __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int i = 0; i < output.length(); i++) {
        // Compute result
        float result = input[warp_index] * weights[warp_index * input.length() + i];

        // Merge sum in warp
        for (int offset = 16; offset > 0; offset /= 2) {
            result += __shfl_sync(FULL_MASK, result, (warp_index + offset) % 32);
        }

        // Set output value
        output[i] = result;
    }
#endif
}

__device__ __host__ void MergeDense::backpropagate() {
#ifdef  __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    input_weights[warp_index] = 0.0;

    for (int i = 0; i < output.length(); i++) {
        input_weights[warp_index] += output_weights[i] * weights[warp_index * input.length() + i];

        delta_weights[warp_index * i] += output_weights[i] * input[warp_index];
    }
#endif
}