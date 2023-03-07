#include "structure.h"

#include <algorithm>
#include <iostream>

__global__ void execute_tasks(StructureTask* task_pointer, bool* task_done_array) {
    int index = blockIdx.x * blockDim.x + threadIdx.x / 32;

    task_pointer[index].perform_task();
    task_done_array[index] = true;
}

Structure::Structure(){}

void Structure::add_tasks(std::vector<StructureTaskWrapper*>* tasks) {}

__host__ __device__ StructureTask::StructureTask() {}

__host__ __device__ StructureTask::StructureTask(Function<float, float>* _layer) {
    layer = _layer;
}

__host__ __device__ void StructureTask::perform_task(){}

StructureTaskWrapper::StructureTaskWrapper() {}

StructureTaskWrapper::StructureTaskWrapper(StructureTask* _task, Function<float, float>* _layer) {
    task = _task;
    layer = _layer;
}

bool StructureTaskWrapper::dependencies_satisfied() {
    if (task_started) return false;

    for (int i = 0; i < dependencies.size(); i++) {
        if (!dependencies[i]->task_done) return false;
    }
    
    return true;
}

StructureScheduler::StructureScheduler(){
    // Reserve GPU memory for tasks
    cudaMalloc(&gpu_tasks, TASK_SIZE * sizeof(StructureTask*));
    cudaMalloc(&gpu_free_spots, TASK_SIZE * sizeof(bool));
}

void StructureScheduler::add_structure(Structure* s) {
    structures.push_back(s);
    s->add_tasks(&task_graph_nodes);
}


StructureTask* StructureScheduler::append_task(StructureTask* task) {
    if (first_task + TASK_SIZE == last_task) throw;

    // Task buffer is a loop buffer
    int head_task = last_task % TASK_SIZE;

    // Copy task to location
    free_spots[head_task] = false;
    tasks[head_task] = task;

    cudaMemcpy(&gpu_free_spots[head_task], &free_spots[head_task], sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&gpu_tasks[head_task], &tasks[head_task], sizeof(StructureTask*), cudaMemcpyHostToDevice);

    // Update task bounds
    last_task++;

    if (task == nullptr) {
        std::cout << "THE TASK IS A FREAKING NULL POINTER" << std::endl;
    }

    return task;
}

__global__ void perform_tasks(StructureTask** tasks, int task_index, bool* free_spots) {
    int warp_offset = threadIdx.x / 32;
    int warp_index = threadIdx.x % 32;
    
    int block_index = blockIdx.x;

    int this_task_index = (task_index + block_index) % TASK_SIZE;

    tasks[this_task_index]->perform_task();
    free_spots[this_task_index] = true;

    // printf("-");
}

void StructureScheduler::start_batch() {
    cudaDeviceSynchronize();

    int chunk_count = 256;

    std::sort(task_graph_nodes.begin(), task_graph_nodes.end(), 
    [](const StructureTaskWrapper* a, const StructureTaskWrapper* b) -> bool {
        return a->distance_to_end > b->distance_to_end;
    });

    for (int i = 0; i < 5; i++) {
        // Reset task counter
        first_task = 0;
        last_task = 0;

        std::vector<StructureTaskWrapper*> running_tasks;

        for (int cycle = 0; cycle < 512; cycle++) {
            int tasks_start = last_task;
            int amount_of_tasks_added = 0;

            // continue;

            for (int i = 0; i < task_graph_nodes.size() && amount_of_tasks_added < chunk_count; i++) {
                StructureTaskWrapper* task = task_graph_nodes[i];

                if (task->dependencies_satisfied()) {                
                    amount_of_tasks_added++;
                    task->task_started = true;
                    running_tasks.push_back(task);
                    StructureTask* t = append_task(task->task);
                    if (t == nullptr) throw "WHAT";
                }
            }
            

            perform_tasks<<<amount_of_tasks_added, 32>>>(gpu_tasks, tasks_start, gpu_free_spots);

            cudaMemcpy(free_spots, gpu_free_spots, TASK_SIZE, cudaMemcpyDeviceToHost);

            int completed_tasks = 0;
            for (int i = first_task; i < last_task; i++) {
                if (free_spots[i % TASK_SIZE]) {
                    completed_tasks++;
                    running_tasks[i % TASK_SIZE]->task_done = true;
                }
            }

            for (int i = first_task; i <= last_task && free_spots[i % TASK_SIZE]; i++) {
                first_task++;
            }

            if (amount_of_tasks_added == 0) break;
        }
        cudaDeviceSynchronize();

        for (int j = 0; j < task_graph_nodes.size(); j++) {
            task_graph_nodes[j]->task_done = false;
            task_graph_nodes[j]->task_started = false;
        }

        std::cout << loss() << std::endl;
    }
}

float StructureScheduler::loss() {
    float* output = new float[structures[0]->output_length];

    cudaMemcpy(output, structures[0]->network_output, sizeof(float) * structures[0]->output_length, cudaMemcpyDeviceToHost);

    float result = 0.0;
    
    for (int i = 0; i < structures[0]->output_length; i++) {
        // std::cout << output[i] << "\t";
        result += output[i];
    }

    // std::cout << std::endl;

    delete output;
    return result;
}
