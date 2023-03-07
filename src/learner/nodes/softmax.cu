#include "softmax.h"
#include <iostream>

__host__ __device__ Softmax::Softmax(int input_size) : Function(0, input_size, input_size) {}

__host__ __device__ Softmax::Softmax(Function* source) : Function(0, source, source->output.length()) {}

__device__ __host__ void Softmax::feed_forward() {
#ifdef  __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    float softmax_sum = 0.0;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        // if (chunk + warp_index < input.length()){
        //     if(input[chunk + warp_index] > 1.0) {
        //         printf("INPUT OF SOFTMAX WAS GREATER THAN 1:, %f\n", input[chunk + warp_index]);
        //         assert(false);
        //     }
        // }

        __syncwarp();

        float softmax_sum_warp = 0.0;
        if (chunk + warp_index < input.length()) {
            softmax_sum_warp = exp(input[chunk + warp_index]);
        }

        // Merge down results of sum
        for (int offset = 16; offset > 0; offset /= 2) {
            softmax_sum_warp += __shfl_sync(FULL_MASK, softmax_sum_warp, (warp_index + offset) % 32);
        }

        softmax_sum += softmax_sum_warp;
    }

    // if (warp_index < output.length()) printf("In: %f \t Out: %f \t Length: %i \n", input[warp_index], output[warp_index], input.length());

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        __syncwarp();
        if (chunk + warp_index < input.length()) {
            output[chunk + warp_index] = exp(input[chunk + warp_index]) / softmax_sum;

            if (isinf(output[chunk + warp_index])) {
                printf("INF value in output of SOFTMAX. Input value was %f \n", input[chunk + warp_index]);
                assert(false);
            }

            if (isnan(output[chunk + warp_index])) {
                printf("nan value in output of SOFTMAX. Input value was %f \n", input[chunk + warp_index]);
                assert(false);
            }
        }
    }
#endif
}

__device__ __host__ void Softmax::backpropagate() {
#ifdef  __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        __syncwarp();

        float softmax_sum_warp = 0.0;
        if (chunk + warp_index < input.length()) {
            input_weights[chunk + warp_index] = output_weights[chunk + warp_index] * output[chunk + warp_index];

            for (int i = 0; i < input.length(); i++) {
                input_weights[chunk + warp_index] -= output_weights[i] * output[chunk + warp_index] * output[i];
            }
        }
    }
#endif
}

__global__ void cuda(Softmax s) {
    s.feed_forward();
    s.backpropagate();
}

void call_softmax_cuda() {
    Softmax s(32);

    s.input[0] = 5.0;
    s.output_weights[0] = 1.0;

    s.move_to_gpu();

    cuda<<<1, 32>>>(s);
    cudaDeviceSynchronize();
}
