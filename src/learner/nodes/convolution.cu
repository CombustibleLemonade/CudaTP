#include "convolution.h"
#include <iostream>

__host__ __device__ Convolution::Convolution(int kernel, int width) : Function(kernel, width, width) {
    _kernel = kernel;
    _width = width;
}

__device__ __host__ void Convolution::feed_forward(){
    // TODO: CUDA warp code
#ifdef  __CUDA_ARCH__

    int warp_index = threadIdx.x % 32;

    // Use the warp index to compute which elements of output should be computed by us
    float* target = &output[warp_index];
    
    // Initialize the target
    *target = 0.0f;

    // Compute the target using the kernel
    for (int i = 0; i < _kernel; i++) {
        int source_idx = (warp_index - i + _width) % _width;
        float weight = weights[i];
        *target += input[source_idx] * weight;
    }

    // Take the sigmoid of the result
    *target = 1/(1+exp(-*target));
#endif
}

__device__ __host__ void Convolution::backpropagate(){
#ifdef  __CUDA_ARCH__
    // Start with output weights
    int warp_index = threadIdx.x % 32;

    // Backpropagate the weights
    for (int i = 0; i < _kernel; i++) {
        int target_idx = (warp_index + i) % _width;
        float weight = weights[i];

        input_weights[warp_index] += output[target_idx] * (1 - output[target_idx]) * weights[i] * output_weights[target_idx];
        
        // Compute delta for parameter
        float delta = output[target_idx] * (1 - output[target_idx]) * input[warp_index] * output_weights[target_idx];

        // Compute sum of delta in warp
        for (int offset = 16; offset > 0; offset /= 2) {
            delta += __shfl_sync(FULL_MASK, delta, (warp_index + offset) % 32);
        }

        delta_weights[i] = delta;
    }

#endif
}

__global__ void perform_convolution(Convolution c) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c.feed_forward();
}

void call_cuda() {
    Convolution c(8, 32);

    c.input[0] = 1.0f;
    // c.input[1] = 1.0f;
    // c.input[3] = 1.0f;
    // c.weights[1] = 1.0f;
    // c.weights[2] = 1.0f;
    // c.weights[3] = 1.5f;

    c.move_to_gpu();

    perform_convolution<<<1, 32>>>(c);
    cudaDeviceSynchronize();
}