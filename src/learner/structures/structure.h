#ifndef STRUCTURE
#define STRUCTURE

#define SCHEDULE_SIZE 1048576
#define TASK_SIZE 65536

#include <vector>
#include "../data/data.h"

class StructureTaskWrapper;

class StructureScheduler;

template <class A, class B> class Function;

class Structure{
public:
    int output_length;
    float* network_output;
    
    StructureScheduler* scheduler;

    Structure();
    virtual void add_tasks(std::vector<StructureTaskWrapper*>* tasks);
};

class StructureTask{
    int index;
public:
    int test_int = 42;

    __host__ __device__ StructureTask();
    __host__ __device__ StructureTask(Function<float, float>* layer);

    Function<float, float>* layer;

    virtual int size() {
        return sizeof(StructureTask);
    }

    __host__ __device__ virtual void perform_task();
};

__global__ void execute_tasks(StructureTask* task_pointer);

class StructureTaskMutex{
public:
    std::vector<StructureTaskWrapper*> dependencies;
    std::vector<StructureTaskWrapper*> dependants;

    int remaining_dependencies;
};

class StructureTaskWrapper{
public:
    StructureTaskWrapper();
    StructureTaskWrapper(StructureTask* task, Function<float, float>* layer);

    std::string info = "Default";

    Function<float, float>* layer;

    int epoch = 0;

    bool task_started = false;
    bool task_done = false;

    int distance_from_start = -1;
    std::vector<StructureTaskWrapper*> dependencies;

    int distance_to_end = -1;
    std::vector<StructureTaskWrapper*> dependants;

    std::vector<StructureTaskMutex*> mutex_dependencies;
    std::vector<StructureTaskMutex*> mutex_dependants;
    
    StructureTask* task;

    bool dependencies_satisfied();
};

class StructureScheduler{
    // The following data is CPU only
    std::vector<Structure*> structures;
    std::vector<StructureTaskWrapper*> task_graph_nodes;

    std::vector<StructureTaskMutex*> task_mutexes;

    // The following data can live on the GPU
    int first_task = 0;
    int last_task = 0;

    StructureTask* tasks[TASK_SIZE]; // CPU
    bool free_spots[TASK_SIZE]; // CPU

    StructureTask** gpu_tasks; // GPU
    bool* gpu_free_spots; // GPU

    Data<char> gpu_memory; // The entire allocated GPU memory, managed by the scheduler

    // __global__ void perform_tasks(StructureTask* task);
public:
    StructureScheduler();

    StructureTask* append_task(StructureTask* task);

    virtual void add_structure(Structure* s);
    virtual void start_batch();
    virtual float loss();
};

template<class T>
void Data<T>::move_to_scheduler(StructureScheduler* scheduler) {
    scheduler->start_batch();
}

#endif