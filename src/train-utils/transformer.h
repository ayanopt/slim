#pragma once
#include "calculations.h"
#include "tokenize.h"
#include <thread>
#include <mutex>
#include <future>
#include <random>

using namespace std;

typedef vector<vector<float>> fmatrix;
typedef vector<float> fvector;

struct TransformerConfig {
    int vocab_size;
    int d_model;
    int n_heads;
    int n_layers;
    int seq_len;
    int d_ff;
    float dropout;
};

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int n_heads);
    fmatrix forward(const fmatrix& input, bool mask = false);
    
private:
    int d_model, n_heads, d_k;
    fmatrix W_q, W_k, W_v, W_o;
    fmatrix scaled_dot_product_attention(const fmatrix& Q, const fmatrix& K, const fmatrix& V, bool mask);
};

class FeedForward {
public:
    FeedForward(int d_model, int d_ff);
    fmatrix forward(const fmatrix& input);
    
private:
    int d_model, d_ff;
    fmatrix W1, W2;
    fvector b1, b2;
};

class TransformerBlock {
public:
    TransformerBlock(int d_model, int n_heads, int d_ff);
    fmatrix forward(const fmatrix& input);
    
private:
    MultiHeadAttention attention;
    FeedForward ff;
    int d_model;
    fmatrix layer_norm(const fmatrix& input);
};

class Transformer {
public:
    Transformer(const TransformerConfig& config);
    fvector forward(const vector<int>& tokens);
    void train(const vector<vector<int>>& batches, int epochs);
    
private:
    TransformerConfig config;
    vector<TransformerBlock> blocks;
    fmatrix token_embedding;
    fmatrix pos_embedding;
    fmatrix output_projection;
    
    fmatrix get_embeddings(const vector<int>& tokens);
    void update_weights(const fmatrix& gradients);
};

fmatrix matrix_multiply_concurrent(const fmatrix& a, const fmatrix& b);
fmatrix softmax(const fmatrix& input);
fmatrix relu(const fmatrix& input);
fmatrix add_matrices(const fmatrix& a, const fmatrix& b);