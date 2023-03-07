#include "data.h"

template <>
__host__ __device__ void Data<float>::initialize(int length, float variance, float bias) {
    printf("asjkldhfgbaiwueydfgiuadwsgyf\n");
    
    _length = length;
    _start = 0;
    _end = length;
    
    source_data = new float[length];

    initialized = true;
}
