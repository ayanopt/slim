#pragma once
#include "layers.h"
#include <atomic>

struct Config {
    int vocab_size = 1000;
    int dim = 256;
    int n_heads = 8;
    int n_layers = 6;
    int seq_len = 256;
    int hidden_dim = 1024;
    float lr = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    float warmup_steps = 100.0f;
    float min_confidence = 0.3f;
};

class Transformer {
public:
    Config cfg;
    Tensor embed, unembed;
    RMSNorm final_norm;
    vector<TransformerBlock> layers;
    RotaryEmbedding rope;
    
    fvec last_hidden;
    fvec embed_input;
    int step = 0;

    Transformer(const Config& c) 
        : cfg(c), final_norm(c.dim), rope(c.dim / c.n_heads, c.seq_len) {
        embed = Tensor(c.vocab_size, c.dim);
        unembed = Tensor(c.dim, c.vocab_size);
        embed.xavier_init();
        unembed.xavier_init();
        
        layers.reserve(c.n_layers);
        for (int i = 0; i < c.n_layers; ++i) {
            layers.emplace_back(c.dim, c.n_heads, c.hidden_dim);
        }
    }

    void resize_vocab(int new_size) {
        if (new_size == cfg.vocab_size) return;
        
        Tensor new_embed(new_size, cfg.dim);
        Tensor new_unembed(cfg.dim, new_size);
        
        int copy_size = min(cfg.vocab_size, new_size);
        for (int i = 0; i < copy_size; ++i) {
            for (int j = 0; j < cfg.dim; ++j) {
                new_embed.at(i, j) = embed.at(i, j);
            }
        }
        for (int i = 0; i < cfg.dim; ++i) {
            for (int j = 0; j < copy_size; ++j) {
                new_unembed.at(i, j) = unembed.at(i, j);
            }
        }
        
        if (new_size > cfg.vocab_size) {
            random_device rd;
            mt19937 gen(rd());
            float std = sqrtf(2.0f / (new_size + cfg.dim));
            normal_distribution<float> dist(0.0f, std);
            for (int i = cfg.vocab_size; i < new_size; ++i) {
                for (int j = 0; j < cfg.dim; ++j) {
                    new_embed.at(i, j) = dist(gen);
                }
            }
            for (int i = 0; i < cfg.dim; ++i) {
                for (int j = cfg.vocab_size; j < new_size; ++j) {
                    new_unembed.at(i, j) = dist(gen);
                }
            }
        }
        
        embed = std::move(new_embed);
        unembed = std::move(new_unembed);
        cfg.vocab_size = new_size;
    }

    fvec forward(const vector<int>& tokens, bool use_cache = false) {
        int seq_len = tokens.size();
        fvec x(seq_len * cfg.dim);
        embed_input.resize(seq_len);
        
        for (int s = 0; s < seq_len; ++s) {
            int tok = tokens[s] % cfg.vocab_size;
            embed_input[s] = tok;
            for (int i = 0; i < cfg.dim; ++i) {
                x[s * cfg.dim + i] = embed.at(tok, i);
            }
        }

        for (auto& layer : layers) {
            x = layer.forward(x, rope, use_cache);
        }

        fvec scales;
        last_hidden = final_norm.forward(x, scales);

        fvec logits(cfg.vocab_size);
        int last = seq_len - 1;
        
        int nt = num_threads();
        vector<thread> threads;
        
        auto compute_logits = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                float val = 0.0f;
                for (int j = 0; j < cfg.dim; ++j) {
                    val += last_hidden[last * cfg.dim + j] * unembed.at(j, i);
                }
                logits[i] = val;
            }
        };

        if (cfg.vocab_size >= nt * 100) {
            int chunk = cfg.vocab_size / nt;
            for (int t = 0; t < nt; ++t) {
                int start = t * chunk;
                int end = (t == nt - 1) ? cfg.vocab_size : start + chunk;
                threads.emplace_back(compute_logits, start, end);
            }
            for (auto& th : threads) th.join();
        } else {
            compute_logits(0, cfg.vocab_size);
        }

        return logits;
    }

    float train_step(const vector<int>& tokens) {
        if (tokens.size() < 2) return 0.0f;
        
        zero_grad();
        
        vector<int> input(tokens.begin(), tokens.end() - 1);
        int target = tokens.back() % cfg.vocab_size;
        
        fvec logits = forward(input, false);
        
        float max_logit = *max_element(logits.begin(), logits.end());
        float sum_exp = 0.0f;
        fvec probs(cfg.vocab_size);
        
        for (int i = 0; i < cfg.vocab_size; ++i) {
            probs[i] = fast_exp(logits[i] - max_logit);
            sum_exp += probs[i];
        }
        sum_exp = max(sum_exp, 1e-10f);
        
        float loss = -logits[target] + max_logit + logf(sum_exp);
        if (!isfinite(loss)) return 10.0f;
        
        fvec grad_logits(cfg.vocab_size);
        for (int i = 0; i < cfg.vocab_size; ++i) {
            grad_logits[i] = probs[i] / sum_exp - (i == target ? 1.0f : 0.0f);
        }
        
        ++step;
        float warmup_factor = min(1.0f, step / cfg.warmup_steps);
        float lr = cfg.lr * warmup_factor;
        
        int seq_len = input.size();
        int last = seq_len - 1;
        
        fvec grad_hidden(seq_len * cfg.dim, 0.0f);
        for (int j = 0; j < cfg.dim; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < cfg.vocab_size; ++i) {
                sum += grad_logits[i] * unembed.at(j, i);
                unembed.grad_at(j, i) += grad_logits[i] * last_hidden[last * cfg.dim + j];
            }
            grad_hidden[last * cfg.dim + j] = sum;
        }
        
        fvec grad_x = grad_hidden;
        for (int l = cfg.n_layers - 1; l >= 0; --l) {
            fvec grad_in;
            layers[l].backward(grad_x, rope, grad_in);
            grad_x = grad_in;
        }
        
        for (int s = 0; s < seq_len; ++s) {
            int tok = static_cast<int>(embed_input[s]);
            for (int i = 0; i < cfg.dim; ++i) {
                embed.grad_at(tok, i) += grad_x[s * cfg.dim + i];
            }
        }
        
        embed.adamw_update(lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, step);
        unembed.adamw_update(lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, step);
        
        for (auto& layer : layers) {
            layer.update(lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, step);
        }
        
        return loss;
    }

    void zero_grad() {
        embed.zero_grad();
        unembed.zero_grad();
        for (auto& layer : layers) layer.zero_grad();
    }

    void clear_cache() {
        for (auto& layer : layers) layer.clear_cache();
    }

    pair<int, float> sample_with_confidence(const fvec& logits, float temperature = 0.8f, float top_p = 0.9f) {
        fvec probs(logits.size());
        float max_l = *max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = fast_exp((logits[i] - max_l) / temperature);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        vector<pair<float, int>> sorted_probs;
        sorted_probs.reserve(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            sorted_probs.emplace_back(probs[i], i);
        }
        sort(sorted_probs.rbegin(), sorted_probs.rend());

        float cumsum = 0.0f;
        size_t cutoff = sorted_probs.size();
        for (size_t i = 0; i < sorted_probs.size(); ++i) {
            cumsum += sorted_probs[i].first;
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }

        float renorm = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) renorm += sorted_probs[i].first;
        
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(0.0f, renorm);
        float r = dist(gen);
        
        cumsum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            cumsum += sorted_probs[i].first;
            if (cumsum >= r) {
                return {sorted_probs[i].second, sorted_probs[0].first};
            }
        }
        return {sorted_probs[0].second, sorted_probs[0].first};
    }

    int sample(const fvec& logits, float temperature = 0.8f, float top_p = 0.9f) {
        return sample_with_confidence(logits, temperature, top_p).first;
    }
};
