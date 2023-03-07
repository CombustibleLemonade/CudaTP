#include "crossentropy.h"

#include <iostream>

__host__ __device__ Crossentropy::Crossentropy(int _size, float* _distribution) : Function(0, _size, 1) {
    distribution_size = _size;
    distribution = _distribution;

    output_weights[0] = 1.0;
}

__host__ __device__ Crossentropy::Crossentropy(Function* source, float* _distribution) : Function(0, source, 1) {
    distribution_size = source->output.length();
    distribution = _distribution;

    output_weights[0] = 1.0;
}

__device__ __host__ void Crossentropy::feed_forward() {
#ifdef __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;
    int block_index = blockIdx.x;

    output[0] = 0.0;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        __syncwarp();

        float result = 0.0;
        if (chunk + warp_index < input.length()) {
            if (isnan(input[chunk + warp_index])) {
                printf("NAN OCCURED IN INPUT OF CROSSENTROPY. \n");
                assert(false);
            }

            // Compute single element cross-entropy
            result = -distribution[chunk + warp_index] * log2(input[chunk + warp_index] + 0.001);
        }

        if (isnan(result)) {
            printf("NAN OCCURED IN OUTPUT OF CROSSENTROPY. Input was %f \n", input[chunk + warp_index]);
            assert(false);
        }

        // if (result > 2.0) printf("%i:%f ", block_index, result);

        // Merge the result into sum
        for (int offset = 16; offset > 0; offset /= 2) {
            result += __shfl_sync(FULL_MASK, result, (warp_index + offset) % 32);
        }

        if (warp_index == 0) {
            output[0] += result;
        }
    }

    // if (warp_index == 0) printf("%i crossentropy is %f\n", block_index, output[0]);

    __syncwarp();
#endif
}

__device__ __host__ void Crossentropy::backpropagate() {
#ifdef __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        __syncwarp();
        
        if (chunk + warp_index < input.length()) {
            float derivative = distribution[chunk + warp_index]/(input[chunk + warp_index] * logf(2));
            
            input_weights[chunk + warp_index] = output_weights[0] * derivative;
        }
    }
#endif
}

void Crossentropy::move_to_gpu() {
    Function<float, float>::move_to_gpu();

    float* gpu_distribution;
    cudaMalloc(&gpu_distribution, sizeof(float) * distribution_size);
    cudaMemcpy(gpu_distribution, distribution, sizeof(float) * distribution_size, cudaMemcpyHostToDevice);
    delete distribution;
    distribution = gpu_distribution;
}