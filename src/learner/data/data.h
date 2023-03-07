#ifndef DATA
#define DATA

#include <iostream>



#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <curand.h>
#include <curand_kernel.h>

class StructureScheduler;

template <class T>
class Data{
    int _start = 0;
    int _end;
    Data* source = nullptr;

    int _length;

    T* source_data;
    bool _is_on_gpu = false;
    bool initialized = false;
public:

    __host__ __device__ Data();
    __host__ __device__ Data(Data* source, int start, int end);
    __host__ __device__ Data(int length);

    __host__ __device__ void initialize(int length, float variance = 1.0, float bias = 0.0);
    
    __device__ __host__ T& operator[](int idx);
    
    __device__ __host__ Data<T> range(int start, int end);

    bool is_on_gpu();
    __device__ __host__ int length();

    void move_to_gpu();
    void copy_to_cpu();

    void move_to_scheduler(StructureScheduler* scheduler);
};

template <class T>
__device__ __host__ Data<T>::Data() {}

template <class T>
__device__ __host__ Data<T>::Data(Data* _source, int start, int end) {
    _start = start;
    _end = end;
    
    if (_source == nullptr) {
        initialize(end - start);
    }
    else {
        source = _source;
        source_data = _source->source_data;
        _start = start;
        _end = end;
        _length = end - start;
    }
}

template <class T>
__device__ __host__ Data<T>::Data(int length) {
    printf("Initialized data of length %i \n", length);
    initialize(length);
}

template <class T>
__device__ __host__ T& Data<T>::operator[](int idx)
{
    #ifdef __CUDA_ARCH__
    if (idx >= _length) {
        printf("Invalid acces at %i in data of size %i \n", idx, _length);
        assert(false);
    }
    #endif
    if (source != nullptr) return (*source)[idx + _start];
    return source_data[idx];
}

template <class T>
__device__ __host__ Data<T> Data<T>::range(int start, int end) {
    if (end > _length) {
        printf("Range is too long. Length is %i, end of range is %i \n", _length, end);
    }

    // #ifdef __CUDA_ARCH__
    // assert(end <= _length);
    // assert(source == nullptr);
    // #endif

    Data<T> result;

    result._length = end - start;
    result.source_data = source_data;

    result._start = start;
    result._end = end;
    result.source = this;

    return result;
}

template <class T>
__host__ __device__ void Data<T>::initialize(int length, float variance, float bias) {
    _length = length;
    _start = 0;
    _end = length;
    
    source_data = new T[length];

    for (int i = 0; i < length; i++) {
        source_data[i] = 0.05;
    }
    source_data[0] = -0.05;

    initialized = true;
}

template <class T>
bool Data<T>::is_on_gpu() {
    return _is_on_gpu;
}

template <class T>
__device__ __host__ int Data<T>::length() {
    return _length;
}

template <class T>
void Data<T>::move_to_gpu() {
    if (_is_on_gpu) return;

    if (source != nullptr) {
        source->move_to_gpu();
        _is_on_gpu = true;
        return;
    }

    if (!initialized) {
        throw;
        return;
    }    

    T* gpu_source_data;
    cudaMalloc(&gpu_source_data, sizeof(T) * _length);
    cudaMemcpy(gpu_source_data, source_data, sizeof(T) * _length, cudaMemcpyHostToDevice);
    delete source_data;
    source_data = gpu_source_data;

    _is_on_gpu = true;
}

template <class T>
void Data<T>::copy_to_cpu() {
    // if (!_is_on_gpu) return;

    T* cpu_source_data = (T*)malloc(sizeof(T)*_length);
    cudaMemcpy(cpu_source_data, source_data, sizeof(T) * _length, cudaMemcpyDeviceToHost);
    // cudaFree(source_data);
    source_data = cpu_source_data;
}

#endif
