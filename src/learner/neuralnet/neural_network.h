#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>
#include <set>

class NeuralNode;

struct NeuralConnection{
    NeuralNode* from;
    NeuralNode* to;

    int index_from;
    int index_to;
    int length;
};

class NeuralNode{
    std::vector<NeuralConnection> incoming_connections;
    std::vector<NeuralConnection> outgoing_connections;

    void connect_back(NeuralNode* source, int index_source, int index_target, int length);
    void connect_forward(NeuralNode* target, int index_source, int index_target, int length);
public:
    void input_from(NeuralNode* source, int index_source=-1, int index_target=-1, int length=-1);
    void output_to(NeuralNode* target, int index_source=-1, int index_target=-1, int length=-1);
};

class NeuralNet{
    std::set<NeuralNode*> nodes;
public:
    bool has_node(NeuralNode* node);
    void add_node(NeuralNode* node);

    float compute_loss();
    float train();
};

#endif