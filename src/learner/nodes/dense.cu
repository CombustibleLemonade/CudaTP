#include "dense.h"

__host__ __device__ Dense::Dense(int input_size, int output_size, Data<float>* _weights, bool _sigmoid_activation) : Function(input_size * output_size, input_size, output_size) {}

__host__ __device__ Dense::Dense(int input_size, int output_size, Function* target, int output_offset, Data<float>* _weights, bool _sigmoid_activation) 
: Function(input_size * output_size + output_size, input_size, output_size, target, output_offset, _weights) {
    bias_index = input_size * output_size;
    sigmoid_activation = _sigmoid_activation;
}

// __host__ __device__ Dense::Dense(int input_size, int output_size, Function* target, int output_offset) 
// : Function(input_size * output_size + output_size, input_size, output_size, target, output_offset) {
//     bias_index = input_size * output_size;
// }

__host__ __device__ Dense::Dense() {}

__device__ __host__ void Dense::feed_forward() {
#ifdef  __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        if (chunk + warp_index < input.length()) {
            assert(!isnan(input[chunk + warp_index]));
            assert(!isinf(input[chunk + warp_index]));
        }
    }

    for (int chunk = 0; chunk < output.length(); chunk += 32) {
        if (chunk + warp_index < output.length()) {
            output[chunk + warp_index] = weights[bias_index + chunk + warp_index];
            
            for (int i = 0; i < input.length(); i++) {
                // Compute result
                float result = input[i] * weights[i * output.length() + chunk + warp_index];

                // Set output value
                output[chunk + warp_index] += result;
            }

            float previous_output = output[chunk + warp_index];

            assert(!isnan(output[chunk + warp_index]));
            assert(!isinf(output[chunk + warp_index]));

            if (sigmoid_activation) output[chunk + warp_index] = 1/(1+exp(-output[chunk + warp_index]));
            
            assert(!isnan(output[chunk + warp_index]));
            assert(!isinf(output[chunk + warp_index]));

            if (sigmoid_activation && output[chunk + warp_index] > 1.0) {
                printf("OUTPUT OF SIGMOID WAS GREATER THAN 1:, %f. Input was %f\n", output[chunk + warp_index], previous_output);
                assert(false);
            }
        }
    }

    // if (warp_index == 0) printf("%f %f \n", input[0], output[0]);
#endif
}

__device__ __host__ void Dense::backpropagate() {
#ifdef  __CUDA_ARCH__
    int warp_index = threadIdx.x % 32;

    // Set all weights to 0
    for (int chunk = 0; chunk < delta_weights.length(); chunk += 32) {
        if (chunk + warp_index < delta_weights.length()) {
            delta_weights[chunk + warp_index] = 0.0;
        }
    }

    for (int chunk = 0; chunk < input.length(); chunk += 32) {
        if (chunk + warp_index < input.length()) {
            input_weights[chunk + warp_index] = 0.0;

            for (int i = 0; i < output.length(); i++) {
                float o = sigmoid_activation ? output[i] * (1 - output[i]) : 1.0;
                float pre_sigmoid_weight = o * output_weights[i];

                if(fabsf(pre_sigmoid_weight) > 1000.0) {
                    printf("Network of size %i x %i has weight %f at index %i. Initialization method was %i \n", 
                    input.length(), output.length(), output_weights[i], i, initialization_method);
                    assert(false);
                }

                if (fabsf(weights[(chunk + warp_index) * output.length() + i]) > 1000.0) {
                    printf("Network of size %i x %i has weight: %f \n", input.length(), output.length(), weights[(chunk + warp_index) * output.length() + i]);
                    assert(false);
                }

                input_weights[chunk + warp_index] += pre_sigmoid_weight * weights[(chunk + warp_index) * output.length() + i];


                delta_weights[(chunk + warp_index) * output.length() + i] = pre_sigmoid_weight * input[chunk + warp_index];
                

                if(fabsf(delta_weights[(chunk + warp_index) * output.length() + i]) > 1000000.0) {
                    printf("Input %f \t output %f \t input_weights %f \t output_weight %f \t weight %f \n", 
                    input[chunk + warp_index], output[i], input_weights[chunk + warp_index], output_weights[i], weights[(chunk + warp_index) * output.length() + i]);
                    
                    assert(false);
                }
            }

            assert(fabsf(input_weights[chunk + warp_index]) < 1000000.0);
        }
    }

    // Weights for bias
    for (int chunk = 0; chunk < output.length(); chunk += 32) {
        if (chunk + warp_index < output.length()) {

            float o =  sigmoid_activation ? output[chunk + warp_index] * (1 - output[chunk + warp_index]) : 1.0;

            // if (output_weights[chunk + warp_index] != 0.0 && output.length() < 100) 
            // printf("o value is %f \n", output[chunk + warp_index] * (1 - output[chunk + warp_index]) ? sigmoid_activation : 1.0);

            delta_weights[bias_index + chunk + warp_index] = o * output_weights[chunk + warp_index];
            
            assert(!isinf(delta_weights[bias_index + chunk + warp_index]));
            assert(!isnan(delta_weights[bias_index + chunk + warp_index]));
        }
    }

    // if (warp_index == 0 && output_weights[0] != 0.0) printf("Weights in: %f \t Weights out: %f \t Input: %f \t Output: %f \t delta: %f \n", input_weights[0], output_weights[0], input[0], output[0], delta_weights[0]);
#endif
}