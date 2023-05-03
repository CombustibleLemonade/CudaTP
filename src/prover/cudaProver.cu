#include "cudaProver.h"

#include <assert.h>
#include <iostream>
#include <ctime>
#include <stdlib.h>
#include <iomanip>

// Hash functions
__host__ __device__ uint64_t cuda_hash(uint64_t i){
	i ^= i >> 33;
	i *= 0xff51afd7ed558ccd;
	i ^= i >> 33;
	i *= 0xc4ceb9fe1a85ec53;
	i ^= i >> 33;
	return i;
}

__host__ __device__ uint64_t cuda_hash2(uint32_t a, uint32_t b){
	uint64_t hash_arg = a;
	hash_arg = hash_arg << 32;
	hash_arg ^= b;
	uint64_t result = cuda_hash(hash_arg);
    return result;
}

__host__ __device__ bool CudaTheorem::is_valid_literal(CudaLiteral* l) {
    if (l->clause_idx < 0) return false;
    if (l->literal_idx < 0) return false;

    if (clause_list.length <= l->clause_idx) return false;
    CudaClause* compare_clause = &clause_list.clauses[l->clause_idx];
    if (compare_clause->length <= l->literal_idx) return false;
    // CudaLiteral* compare_literal = &compare_clause->literals[l->literal_idx];
    
    return true;
}

__host__ bool CudaTheorem::fill_matrix() {
    CudaProofState proof_state;
    proof_state.theorem = this;

    int list_at = 0;

    for (int c1 = 0; c1 < clause_list.length; c1++) {
        for (int l1 = 0; l1 < clause_list.clauses[c1].length; l1++) {
            CudaLiteral* literal1 = &clause_list.clauses[c1].literals[l1];
            CudaMatrixIndex* matrix_index = &matrix.indices[c1 * CLAUSE_WIDTH + l1];
            
            matrix_index->length = 0;
            matrix_index->index = list_at;

            for (int c2 = 0; c2 < clause_list.length; c2++) {
                for (int l2 = 0; l2 < clause_list.clauses[c2].length; l2++) {
                    CudaLiteral* literal2 = &clause_list.clauses[c2].literals[l2];
                    
                    if (proof_state.unify(literal1, 0, literal2, 1)) {
                        matrix_index->length++;
                        matrix.list[list_at++] = c2 * CLAUSE_WIDTH + l2;
                        proof_state.mutation.table.pop();
                    }
                }
            }
        }
    }

    for (int c = 0; c < clause_list.length; c++) {
        for (int l = 0; l < clause_list.clauses[c].length; l++) {
            CudaMatrixIndex* matrix_index = &matrix.indices[c * CLAUSE_WIDTH + l];
            float total = 0;

            for (int i = 0; i < matrix_index->length; i++) {
                int index = i + matrix_index->index;
                CudaMatrixIndex* target = &matrix.indices[matrix.list[index]];

                CudaClause* to_clause = &clause_list.clauses[target->index / CLAUSE_WIDTH];
                CudaLiteral* to_literal = &to_clause->literals[target->index % CLAUSE_WIDTH];

                // float score = target->length;
                // float score = to_clause->length;
                float score = 1.0 / to_clause->length;

                // float score = 1.0;

                total += score;
                matrix.probabilities[index] = score;
            }

            
            for (int i = 0; i < matrix_index->length; i++) {
                int index = i + matrix_index->index;
                matrix.probabilities[index] /= total;
            }
        }
    }

    return true;
}

__host__ __device__ bool CudaSubstitution::occupied() {
    return from_node != -1;
}

__host__ __device__ void CudaSubstitution::erase() {
    from_node = -1;
}

__host__ __device__ CudaSubstitutionTable::CudaSubstitutionTable() {
    // Populate the table with the initial node
    u_int64_t idx = cuda_hash2(0, 0);
    u_int computed_idx = idx % CUDA_SUBSTITUTION_TABLE_SIZE;
    head = &substitutions[computed_idx];

    head->from_node = 0;
    head->from_sub_idx = 0;

    element_count++;
}

__host__ __device__ CudaSubstitution* CudaSubstitutionTable::append(int node, int substitution_idx) {
    u_int64_t idx = cuda_hash2(node, substitution_idx);
    u_int computed_idx = idx % CUDA_SUBSTITUTION_TABLE_SIZE;
    CudaSubstitution* sub = &substitutions[computed_idx];

    while (sub->occupied()) {
        idx = cuda_hash(idx);
        computed_idx = idx % CUDA_SUBSTITUTION_TABLE_SIZE;
        sub = &substitutions[computed_idx];
    }

    if (head->from_node == 0) {
        assert(substitution_idx <= head->from_sub_idx + 1);
    }

    sub->from_node = node;
    sub->from_sub_idx = substitution_idx;
    sub->previous = head;
    head = sub;

    element_count++;

    return sub;
}

__host__ __device__ CudaSubstitution* CudaSubstitutionTable::get(int node, int substitution_idx) {
    CudaSubstitution* result = nullptr;
    CudaSubstitution* sub = nullptr;

    while (true) {
        u_int64_t idx = cuda_hash2(node, substitution_idx);
        u_int computed_idx = idx % CUDA_SUBSTITUTION_TABLE_SIZE;
        sub = &substitutions[computed_idx];

        while (sub->occupied()) {
            if (sub->from_node == node && sub->from_sub_idx == substitution_idx) {
                node = sub->to_node;
                substitution_idx = sub->to_sub_idx;

                result = sub;
                break;
            }

            idx = cuda_hash(idx);
            computed_idx = idx % CUDA_SUBSTITUTION_TABLE_SIZE;
            sub = &substitutions[computed_idx];
        }

        if (!sub->occupied()) break;
    }

    return result;
}

__host__ __device__ void CudaSubstitutionTable::push(int substitution_idx) {
    append(0, substitution_idx);
}

__host__ __device__ void CudaSubstitutionTable::pop() {
    decapitate();
    while (head->from_node != 0) {
        decapitate();
    }
}

__host__ __device__ void CudaSubstitutionTable::decapitate(){
    CudaSubstitution* neck = head->previous;
    head->erase();
    head = neck;

    element_count--;
}

__host__ __device__ void CudaActionLog::undo() {}

__host__ __device__ void CudaActionLog::register_action(CudaActionLogEntry action) {
    actions[action_length++] = action;
    if (action.type == RETRACT) {
        stack_length--;
    } else if (action.type == EXTEND) {
        stack[stack_length++] = action;
    }
}

__host__ __device__ void CudaActionLog::backtrack(){}

__host__ __device__ void CudaMutation::tread_back() {
    while (visited == 255) {
        if (log.stack_length == 0) {
            solved = true;
            return;
        }

        // assert(log.stack_length < 64);
        // assert(log.action_length < 128);

        CudaActionLogEntry entry = log.stack[log.stack_length - 1];
        entry.type = RETRACT;
        visited = entry.visit_log | (1 << entry.source->literal_idx);
        head = entry.source;
        log.register_action(entry);
    }
}

__host__ __device__ int CudaProofState::get_random_int(int min, int max) {
    assert(min < max);
    
    #ifdef __CUDA_ARCH__
    float r = curand_uniform(random);
    float rand_f = r * (max - min) + min;
    int result = __float2int_rd(rand_f);
    
    // Clamp value for rounding error edge cases
    result = result < min ? min : result;
    result = result >= max ? max - 1 : result;

    return result;

    #else
    return (rand() % (max - min)) + min;
    #endif
}

__host__ __device__ int CudaProofState::pick_next_literal() {
    int valid_literal_count = 0;
    int valid_literals[8];
    
    for (int i = 0; i < CLAUSE_WIDTH; i++) {
        int mask = 1 << i;
        if ( (mutation.visited & mask) == 0) {
            valid_literals[valid_literal_count++] = i;
        }
    }

    return valid_literals[0];
}

__host__ __device__ bool CudaProofState::occurs(CudaSubstitution sub) {
    int stack_idx = 1;
    occurs_stack[0] = sub;

    int occurs_breath = 0;

    while (stack_idx-- > 0) {
        occurs_breath++;
        if (occurs_breath > 1024) {
            occurs_overflows++;
            return true;
        }
        sub = occurs_stack[stack_idx];
        if (sub.from_node == sub.to_node && sub.from_sub_idx == sub.to_sub_idx) return true;

        if ((sub.to_node & NODE_TYPE_MASK) == VARIABLE_MASK) {
            CudaSubstitution* s = mutation.table.get(sub.to_node, sub.to_sub_idx);
            if (s != nullptr) {
                sub.to_node = s->to_node;
                sub.to_sub_idx = s->to_sub_idx;
            }
        }

        if (sub.from_node == sub.to_node && sub.from_sub_idx == sub.to_sub_idx) return true;

        if ((sub.to_node & NODE_TYPE_MASK) == PAIR_MASK) {
            if (stack_idx >= MAX_OCCURS_STACK_SIZE - 2) {
                printf("Occurs stack exhausted!\n");
                return true;
            }

            CudaPair pair = theorem->pair_list.pair_list[sub.to_node & ~NODE_TYPE_MASK];
            sub.to_node = pair.a;
            occurs_stack[stack_idx++] = sub;
            sub.to_node = pair.b;
            occurs_stack[stack_idx++] = sub;
        }
        assert(stack_idx < MAX_OCCURS_STACK_SIZE);
    }
    return false;
}

__host__ __device__ bool CudaProofState::unify(int a, int a_idx, int b, int b_idx) {
    int stack_idx = 1;

    unify_stack[0].from_node = a;
    unify_stack[0].from_sub_idx = a_idx;
    unify_stack[0].to_node = b;
    unify_stack[0].to_sub_idx = b_idx;

    while (stack_idx > 0) {
        stack_idx--;
        if (stack_idx >= MAX_OCCURS_STACK_SIZE) {
            return false;
        }

        CudaSubstitution sub = unify_stack[stack_idx];

        CudaSubstitution* sub_a = mutation.table.get(sub.from_node, sub.from_sub_idx);
        CudaSubstitution* sub_b = mutation.table.get(sub.to_node, sub.to_sub_idx);

        if (sub_a != nullptr) {
            sub.from_node = sub_a->to_node;
            sub.from_sub_idx = sub_a->to_sub_idx;
        }

        if (sub_b != nullptr) {
            sub.to_node = sub_b->to_node;
            sub.to_sub_idx = sub_b->to_sub_idx;
        }


        if ((sub.to_node & NODE_TYPE_MASK) == VARIABLE_MASK && (sub.from_node & NODE_TYPE_MASK) != VARIABLE_MASK) {
            // Swap Variables
            int buffer = sub.to_node;
            sub.to_node = sub.from_node;
            sub.from_node = buffer;

            buffer = sub.to_sub_idx;
            sub.to_sub_idx = sub.from_sub_idx;
            sub.from_sub_idx = buffer;
        }

        if (sub.from_node == sub.to_node && (sub.from_sub_idx == sub.to_sub_idx || (sub.from_node & NODE_TYPE_MASK) == CONSTANT_MASK)) continue;

        if ((sub.from_node & NODE_TYPE_MASK) == VARIABLE_MASK) {
            if (occurs(sub)) return false;

            CudaSubstitution* var_sub = mutation.table.append(sub.from_node, sub.from_sub_idx);
            var_sub->to_node = sub.to_node;
            var_sub->to_sub_idx = sub.to_sub_idx;
            continue;
        }

        if ((sub.from_node & NODE_TYPE_MASK) == PAIR_MASK && (sub.to_node & NODE_TYPE_MASK) == PAIR_MASK) {
            int pair_idx_a = sub.from_node & ~NODE_TYPE_MASK;
            int pair_idx_b = sub.to_node & ~NODE_TYPE_MASK;

            CudaPair p_a = theorem->pair_list.pair_list[pair_idx_a];
            CudaPair p_b = theorem->pair_list.pair_list[pair_idx_b];

            unify_stack[stack_idx].from_node = p_a.a;
            unify_stack[stack_idx].from_sub_idx = sub.from_sub_idx;
            unify_stack[stack_idx].to_node = p_b.a;
            unify_stack[stack_idx].to_sub_idx = sub.to_sub_idx;

            stack_idx++;

            unify_stack[stack_idx].from_node = p_a.b;
            unify_stack[stack_idx].from_sub_idx = sub.from_sub_idx;
            unify_stack[stack_idx].to_node = p_b.b;
            unify_stack[stack_idx].to_sub_idx = sub.to_sub_idx;

            stack_idx++;
            continue;
        }
        return false;
    }

    return true;
}

__host__ __device__ bool CudaProofState::unify(CudaLiteral* a, int idx_a, CudaLiteral* b, int idx_b) {
    if (a->negative == b->negative) return false;
    if (unify(a->node, idx_a, b->node, idx_b)){
        assert(idx_b >= 0);
        mutation.table.push(mutation.substitution_index);
        return true;
    } else {
        mutation.table.push(mutation.substitution_index);
        mutation.table.pop();
        return false;
    }
}

__host__ __device__ void CudaProofState::register_dead_clauses(){
    mutation.visited = 0;

    for (int i = theorem->clause_list.clauses[mutation.head->clause_idx].length; i < CLAUSE_WIDTH; i++) {
        mutation.visited |= (1 << i);
    }
}

__host__ __device__ bool CudaProofState::extend(CudaLiteral* l) {

    int from_substitution_idx = 0;

    if (mutation.log.stack_length > 0) {
        from_substitution_idx = mutation.log.stack[mutation.log.stack_length - 1].substitution_index;
    }

    if (!unify(mutation.head, from_substitution_idx, l, mutation.substitution_index)) {
        return false;
    }

    // #ifndef __CUDA_ARCH__
    // std::string eq_a = to_string(mutation.head->node, from_substitution_idx);
    // std::string eq_b = to_string(l->node, mutation.substitution_index);
    // bool equivalent = (eq_a == eq_b);
    // assert(equivalent);
    // #endif

    CudaActionLogEntry action;
    action.type = EXTEND;
    action.source = mutation.head;
    action.target = l;
    action.visit_log = mutation.visited;
    action.substitution_index = mutation.substitution_index++;

    mutation.log.register_action(action);
    mutation.head = l;

    register_dead_clauses();
    mutation.visited |= (1 << l->literal_idx);
    mutation.tread_back();
    int literal = pick_next_literal();
    mutation.head = &theorem->clause_list.clauses[mutation.head->clause_idx].literals[literal];

    inferences++;

    return true;
}

__host__ __device__ bool CudaProofState::reduce(int idx) {
    CudaActionLogEntry past = mutation.log.stack[idx];
    CudaLiteral* to = past.source;
    int sub_idx = 0;
    if (idx > 0) {
        sub_idx = mutation.log.stack[idx-1].substitution_index;
    }

    if (!unify(to, sub_idx, mutation.head, mutation.log.stack[mutation.log.stack_length - 1].substitution_index)) {
        return false;
    }

    CudaActionLogEntry entry;
    entry.type = REDUCE;
    entry.source = mutation.head;
    entry.target = to;
    entry.substitution_index = past.substitution_index;
    entry.visit_log = mutation.visited;
    
    mutation.substitution_index++;
    mutation.log.register_action(entry);

    mutation.visited |= (1 << mutation.head->literal_idx);
    mutation.tread_back();

    int literal = pick_next_literal();
    mutation.head = &theorem->clause_list.clauses[mutation.head->clause_idx].literals[literal];

    return true;
}

__host__ __device__ void CudaProofState::undo() {
    assert(mutation.log.action_length > 0);

    while (mutation.log.actions[--mutation.log.action_length].type == RETRACT) {
        // Reappend to the stack
        CudaActionLogEntry retraction = mutation.log.actions[mutation.log.action_length];
        retraction.type = EXTEND;
        mutation.log.stack[mutation.log.stack_length++] = retraction;
    }

    CudaActionLogEntry action = mutation.log.actions[mutation.log.action_length];
    mutation.head = action.source;
    mutation.substitution_index = action.substitution_index;
    mutation.visited = action.visit_log;
    mutation.table.pop();

    if (action.type == EXTEND) {
        mutation.log.stack_length--;
    }

    mutation.solved = false;
}

__host__ __device__ CudaLiteral* CudaProofState::get_random_literal(){
    #ifdef __CUDA_ARCH__
    float rand_f = 1 - curand_uniform(random);
    int clause_idx = __float2int_rd(rand_f * theorem->clause_list.length) % theorem->clause_list.length;

    rand_f = 1 - curand_uniform(random);
    int literal_idx = __float2int_rd(rand_f * theorem->clause_list.clauses[clause_idx].length) % theorem->clause_list.clauses[clause_idx].length;
    
    #else
    int clause_idx = rand() % theorem->clause_list.length;
    int literal_idx = rand() % theorem->clause_list.clauses[clause_idx].length;
    #endif

    CudaLiteral* result = &theorem->clause_list.clauses[clause_idx].literals[literal_idx];

    return result;
}

__host__ __device__ bool CudaProofState::prove() {
    // Pick a random theorem to use as the head
    mutation.head = get_random_literal();
    register_dead_clauses();
    
    // Proof loop
    for (int i = 0; i < 64 && !mutation.solved; i++) {

        int reduce_to = get_random_int(0, mutation.log.stack_length + 5);
        
        assert(reduce_to >=0);
        assert(reduce_to < mutation.log.stack_length + 20);
        
        if (reduce_to < mutation.log.stack_length) {
            // reduce
            bool reduced = reduce(reduce_to);
        } else {
            int head_idx = mutation.head->clause_idx * CLAUSE_WIDTH + mutation.head->literal_idx;
            CudaMatrixIndex m_idx = theorem->matrix.indices[head_idx];

            // TODO: filter out theorems with ununifiable literals
            if (m_idx.length == 0) {
                return false;
            }

            if (m_idx.length <= 0) {
                printf("Length: %i, head_idx %i \n", m_idx.length, head_idx);
            }

            assert(m_idx.length > 0);

            // extend
            reduce_to = get_random_int(0, m_idx.length);
            assert(reduce_to >= 0);
            assert(reduce_to < m_idx.length);
            reduce_to += m_idx.index;


        #ifdef __CUDA_ARCH__
            float r = curand_uniform(random);
        #else
            float r = 0.0; // TODO
        #endif

            for (int j = 0; j < m_idx.length; j++){
                r -= theorem->matrix.probabilities[m_idx.index + j];
                if (r <= 0) {
                    reduce_to = m_idx.index + j;
                    break;
                }
            }

            int index = theorem->matrix.list[reduce_to];
            int literal_index = index % CLAUSE_WIDTH;
            int clause_index = index / CLAUSE_WIDTH;

            CudaLiteral* to = &theorem->clause_list.clauses[clause_index].literals[literal_index]; 
            // CudaLiteral* from = mutation.head;

            // CudaLiteral* to = get_random_literal();
            bool extended = extend(to);
        }
    }

    return mutation.solved;
}

__device__ void CudaProofState::hash_test() {
    for (int i = 0; i < 2048; i++) {

        float rand_f = 1 - curand_uniform(random);
        int a = __float2int_rd(rand_f * 16384);

        rand_f = 1 - curand_uniform(random);
        int b = __float2int_rd(rand_f * 16384);
        if (!mutation.table.get(a, b)) {
            CudaSubstitution* s = mutation.table.append(a, b);
        }
    }

    mutation.table.push(1);
    mutation.table.pop();
}

__host__ std::string CudaProofState::to_string(int node, int sub_idx, int depth) {
    assert(depth < 128);

    CudaSubstitution* sub;
    if ((node & NODE_TYPE_MASK) == VARIABLE_MASK){
        sub = mutation.table.get(node, sub_idx);

        if (sub != nullptr) {
            node = sub->to_node;
            sub_idx = sub->to_sub_idx;
        }
    }

    if ((node & NODE_TYPE_MASK) == PAIR_MASK) {
        CudaPair pair = theorem->pair_list.pair_list[node & ~NODE_TYPE_MASK];
        std::string result = "(";
        result += to_string(pair.a, sub_idx, depth + 1);
        result += ", ";
        result += to_string(pair.b, sub_idx, depth + 1);
        result += ")";

        return result;
    }

    std::stringstream stream;
    stream << std::hex << node;
    return std::string(stream.str());
}

__global__ void proof_thread(CudaTheorem* t, curandState* state_list, CudaProofState* proof_state, int* any_solved, Solution* solution) {    
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    curandState* state = &state_list[id];
    curand_init(clock64(), id, 0, state);

    CudaProofState* p = &proof_state[id]; //new CudaMutation;
    any_solved = &any_solved[id];
    *any_solved = 0;
    solution = &solution[id];

    new((void*)p) CudaProofState();

    p->theorem = t;
    p->random = state;

    for (int i = 0; i < 3200 && p != nullptr; i++) {
        bool solved = p->prove();

        if (solved) {
            *any_solved += 1;

            solution->length = p->mutation.log.action_length;

            for (int j = 0; j < solution->length; j++){
                // if (p->mutation.log.actions[j].type == CudaActionType::RETRACT) continue;
                solution->steps[j].step_type = p->mutation.log.actions[j].type;

                solution->steps[j].from_clause_idx = p->mutation.log.actions[j].source->clause_idx;
                solution->steps[j].from_literal_idx = p->mutation.log.actions[j].source->literal_idx;

                solution->steps[j].to_clause_idx = p->mutation.log.actions[j].target->clause_idx;
                solution->steps[j].to_literal_idx = p->mutation.log.actions[j].target->literal_idx;
            }
        }

        while (p->mutation.log.action_length != 0) {
            p->undo();
        }
    }

    printf("Inferences: %i \n", p->inferences);
}

ComputationResult* start_prover(CudaTheorem* t) {
	clock_t start = clock();
    int* gpu_solved;
    cudaMalloc(&gpu_solved, sizeof(int) * BLOCK_COUNT * THREAD_PER_BLOCK);

    curandState* state = new curandState[BLOCK_COUNT * THREAD_PER_BLOCK];
    curandState* device_state;
    cudaMalloc(&device_state, sizeof(curandState) * BLOCK_COUNT * THREAD_PER_BLOCK);
    cudaMemcpy(device_state, state, sizeof(curandState) * BLOCK_COUNT * THREAD_PER_BLOCK, cudaMemcpyHostToDevice);
    delete state;

    CudaProofState* proof_state;
    cudaMalloc(&proof_state, sizeof(CudaProofState) * BLOCK_COUNT * THREAD_PER_BLOCK);

    // Move proofstate to gpu
    CudaTheorem* t_gpu;
    cudaMalloc((void**)&t_gpu, sizeof(CudaTheorem));
    cudaMemcpy(t_gpu, t, sizeof(CudaTheorem), cudaMemcpyHostToDevice);

    Solution* gpu_solutions;
    cudaMalloc(&gpu_solutions, sizeof(Solution)*BLOCK_COUNT*THREAD_PER_BLOCK);

    proof_thread<<<BLOCK_COUNT, THREAD_PER_BLOCK>>>(t_gpu, device_state, proof_state, gpu_solved, gpu_solutions);

    cudaDeviceSynchronize();
	clock_t end = clock();

    float seconds = (end - start) /  (float)CLOCKS_PER_SEC;
    // std::cout << "Completed task in " << seconds << " seconds." << std::endl;

    ComputationResult* result = new ComputationResult;
    int* cpu_solved = (int*)malloc(sizeof(int) * BLOCK_COUNT * THREAD_PER_BLOCK);
    cudaMemcpy(&result->solution_counts, gpu_solved, sizeof(int) * BLOCK_COUNT * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);

    cudaMemcpy(&result->solutions, gpu_solutions, sizeof(Solution) * BLOCK_COUNT * THREAD_PER_BLOCK, cudaMemcpyDeviceToHost);

    cudaDeviceReset();

    result->total_solutions = 0;

    int total = 0;
    for (int i = 0; i < BLOCK_COUNT * THREAD_PER_BLOCK; i++) {
        result->total_solutions += result->solution_counts[i];
    }

    // Deallocate pointers
    free(cpu_solved);

    return result;
}