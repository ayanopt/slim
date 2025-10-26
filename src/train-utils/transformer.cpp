#include "transformer.h"
#include <cmath>
#include <algorithm>

fmatrix matrix_multiply_concurrent(const fmatrix& a, const fmatrix& b) {
    size_t rows_a = a.size();
    size_t cols_a = a[0].size();
    size_t cols_b = b[0].size();
    
    fmatrix result(rows_a, fvector(cols_b, 0.0f));
    
    auto worker = [&](size_t start_row, size_t end_row) {
        for (size_t i = start_row; i < end_row; ++i) {
            for (size_t j = 0; j < cols_b; ++j) {
                for (size_t k = 0; k < cols_a; ++k) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    };
    
    size_t n_threads = thread::hardware_concurrency();
    size_t chunk_size = rows_a / n_threads;
    vector<thread> threads;
    
    for (size_t t = 0; t < n_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == n_threads - 1) ? rows_a : start + chunk_size;
        threads.emplace_back(worker, start, end);
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    return result;
}

fmatrix softmax(const fmatrix& input) {
    fmatrix result = input;
    for (auto& row : result) {
        float max_val = *max_element(row.begin(), row.end());
        float sum = 0.0f;
        for (auto& val : row) {
            val = exp(val - max_val);
            sum += val;
        }
        for (auto& val : row) {
            val /= sum;
        }
    }
    return result;
}

fmatrix relu(const fmatrix& input) {
    fmatrix result = input;
    for (auto& row : result) {
        for (auto& val : row) {
            val = max(0.0f, val);
        }
    }
    return result;
}

fmatrix add_matrices(const fmatrix& a, const fmatrix& b) {
    fmatrix result(a.size(), fvector(a[0].size()));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

MultiHeadAttention::MultiHeadAttention(int d_model, int n_heads) 
    : d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, sqrt(2.0f / d_model));
    
    W_q = fmatrix(d_model, fvector(d_model));
    W_k = fmatrix(d_model, fvector(d_model));
    W_v = fmatrix(d_model, fvector(d_model));
    W_o = fmatrix(d_model, fvector(d_model));
    
    for (auto& matrix : {&W_q, &W_k, &W_v, &W_o}) {
        for (auto& row : *matrix) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
    }
}

fmatrix MultiHeadAttention::scaled_dot_product_attention(const fmatrix& Q, const fmatrix& K, const fmatrix& V, bool mask) {
    fmatrix K_T(K[0].size(), fvector(K.size()));
    for (size_t i = 0; i < K.size(); ++i) {
        for (size_t j = 0; j < K[0].size(); ++j) {
            K_T[j][i] = K[i][j];
        }
    }
    
    fmatrix scores = matrix_multiply_concurrent(Q, K_T);
    float scale = 1.0f / sqrt(d_k);
    
    for (auto& row : scores) {
        for (auto& val : row) {
            val *= scale;
        }
    }
    
    if (mask) {
        for (size_t i = 0; i < scores.size(); ++i) {
            for (size_t j = i + 1; j < scores[0].size(); ++j) {
                scores[i][j] = -1e9f;
            }
        }
    }
    
    fmatrix attention_weights = softmax(scores);
    return matrix_multiply_concurrent(attention_weights, V);
}

fmatrix MultiHeadAttention::forward(const fmatrix& input, bool mask) {
    fmatrix Q = matrix_multiply_concurrent(input, W_q);
    fmatrix K = matrix_multiply_concurrent(input, W_k);
    fmatrix V = matrix_multiply_concurrent(input, W_v);
    
    vector<fmatrix> head_outputs;
    
    for (int h = 0; h < n_heads; ++h) {
        int start_col = h * d_k;
        int end_col = start_col + d_k;
        
        fmatrix Q_h(Q.size(), fvector(d_k));
        fmatrix K_h(K.size(), fvector(d_k));
        fmatrix V_h(V.size(), fvector(d_k));
        
        for (size_t i = 0; i < Q.size(); ++i) {
            for (int j = 0; j < d_k; ++j) {
                Q_h[i][j] = Q[i][start_col + j];
                K_h[i][j] = K[i][start_col + j];
                V_h[i][j] = V[i][start_col + j];
            }
        }
        
        head_outputs.push_back(scaled_dot_product_attention(Q_h, K_h, V_h, mask));
    }
    
    fmatrix concat(input.size(), fvector(d_model));
    for (size_t i = 0; i < input.size(); ++i) {
        for (int h = 0; h < n_heads; ++h) {
            for (int j = 0; j < d_k; ++j) {
                concat[i][h * d_k + j] = head_outputs[h][i][j];
            }
        }
    }
    
    return matrix_multiply_concurrent(concat, W_o);
}

FeedForward::FeedForward(int d_model, int d_ff) : d_model(d_model), d_ff(d_ff) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0f, sqrt(2.0f / d_model));
    
    W1 = fmatrix(d_model, fvector(d_ff));
    W2 = fmatrix(d_ff, fvector(d_model));
    b1 = fvector(d_ff, 0.0f);
    b2 = fvector(d_model, 0.0f);
    
    for (auto& row : W1) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
    for (auto& row : W2) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
}

fmatrix FeedForward::forward(const fmatrix& input) {
    fmatrix hidden = matrix_multiply_concurrent(input, W1);
    
    for (size_t i = 0; i < hidden.size(); ++i) {
        for (size_t j = 0; j < hidden[0].size(); ++j) {
            hidden[i][j] += b1[j];
        }
    }
    
    hidden = relu(hidden);
    fmatrix output = matrix_multiply_concurrent(hidden, W2);
    
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[0].size(); ++j) {
            output[i][j] += b2[j];
        }
    }
    
    return output;
}

TransformerBlock::TransformerBlock(int d_model, int n_heads, int d_ff) 
    : attention(d_model, n_heads), ff(d_model, d_ff), d_model(d_model) {}

fmatrix TransformerBlock::layer_norm(const fmatrix& input) {
    fmatrix result = input;
    for (auto& row : result) {
        float mean = 0.0f;
        for (auto val : row) {
            mean += val;
        }
        mean /= row.size();
        
        float variance = 0.0f;
        for (auto val : row) {
            variance += (val - mean) * (val - mean);
        }
        variance /= row.size();
        
        float std_dev = sqrt(variance + 1e-6f);
        for (auto& val : row) {
            val = (val - mean) / std_dev;
        }
    }
    return result;
}

fmatrix TransformerBlock::forward(const fmatrix& input) {
    fmatrix attn_output = attention.forward(input, true);
    fmatrix norm1 = layer_norm(add_matrices(input, attn_output));
    
    fmatrix ff_output = ff.forward(norm1);
    return layer_norm(add_matrices(norm1, ff_output));
}

Transformer::Transformer(const TransformerConfig& config) : config(config) {
    random_device rd;
    mt19937 gen(rd());
    
    for (int i = 0; i < config.n_layers; ++i) {
        blocks.emplace_back(config.d_model, config.n_heads, config.d_ff);
    }
    
    token_embedding = fmatrix(config.vocab_size, fvector(config.d_model));
    pos_embedding = fmatrix(config.seq_len, fvector(config.d_model));
    output_projection = fmatrix(config.d_model, fvector(config.vocab_size));
    
    // Xavier initialization for better training
    normal_distribution<float> embed_dist(0.0f, sqrt(1.0f / config.d_model));
    normal_distribution<float> proj_dist(0.0f, sqrt(1.0f / config.vocab_size));
    
    for (auto& row : token_embedding) {
        for (auto& val : row) {
            val = embed_dist(gen);
        }
    }
    
    for (auto& row : pos_embedding) {
        for (auto& val : row) {
            val = embed_dist(gen) * 0.1f;
        }
    }
    
    for (auto& row : output_projection) {
        for (auto& val : row) {
            val = proj_dist(gen);
        }
    }
}

fmatrix Transformer::get_embeddings(const vector<int>& tokens) {
    fmatrix embeddings(tokens.size(), fvector(config.d_model));
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        int token_id = tokens[i] % config.vocab_size;
        size_t pos_id = min(i, static_cast<size_t>(config.seq_len - 1));
        
        for (int j = 0; j < config.d_model; ++j) {
            embeddings[i][j] = token_embedding[token_id][j] + pos_embedding[pos_id][j];
        }
    }
    
    return embeddings;
}

fvector Transformer::forward(const vector<int>& tokens) {
    fmatrix x = get_embeddings(tokens);
    
    for (auto& block : blocks) {
        x = block.forward(x);
    }
    
    fmatrix logits = matrix_multiply_concurrent(x, output_projection);
    return logits.back();
}

void Transformer::train(const vector<vector<int>>& batches, int epochs) {
    const float learning_rate = 0.001f;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        vector<future<float>> futures;
        
        for (const auto& batch : batches) {
            futures.push_back(async(launch::async, [this, &batch, learning_rate]() -> float {
                float batch_loss = 0.0f;
                
                for (size_t i = 1; i < batch.size(); ++i) {
                    vector<int> input(batch.begin(), batch.begin() + i);
                    int target = batch[i] % config.vocab_size;
                    
                    fvector logits = forward(input);
                    
                    // Compute cross-entropy loss
                    float max_logit = *max_element(logits.begin(), logits.end());
                    float sum_exp = 0.0f;
                    for (auto& logit : logits) {
                        sum_exp += exp(logit - max_logit);
                    }
                    float loss = -logits[target] + max_logit + log(sum_exp);
                    batch_loss += loss;
                    
                    // Simple gradient update for output layer
                    fvector probs(logits.size());
                    for (size_t j = 0; j < logits.size(); ++j) {
                        probs[j] = exp(logits[j] - max_logit) / sum_exp;
                        if (j == static_cast<size_t>(target)) probs[j] -= 1.0f;
                    }
                    
                    // Update output projection weights
                    for (int j = 0; j < config.d_model; ++j) {
                        for (size_t k = 0; k < probs.size(); ++k) {
                            output_projection[j][k] -= learning_rate * probs[k] * 0.1f;
                        }
                    }
                }
                
                return batch_loss / batch.size();
            }));
        }
        
        float total_loss = 0.0f;
        for (auto& future : futures) {
            total_loss += future.get();
        }
        
        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << total_loss / batches.size() << endl;
        }
    }
}