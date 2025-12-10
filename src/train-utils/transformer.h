#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>

using namespace std;

using fvec = vector<float>;
using fmat = vector<fvec>;

struct Config {
    int vocab_size = 1000;
    int dim = 128;
    int n_heads = 4;
    int n_layers = 4;
    int seq_len = 128;
    int hidden_dim = 512;
    float dropout = 0.1f;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
};

inline float fast_exp(float x) {
    x = 1.0f + x / 256.0f;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

inline float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
    return x / (1.0f + fast_exp(-x));
}

class Tensor {
public:
    fvec data;
    fvec grad;
    fvec m, v;
    int rows, cols;

    Tensor() : rows(0), cols(0) {}
    Tensor(int r, int c, bool init_adam = true) : rows(r), cols(c) {
        data.resize(r * c);
        grad.resize(r * c, 0.0f);
        if (init_adam) {
            m.resize(r * c, 0.0f);
            v.resize(r * c, 0.0f);
        }
    }

    void xavier_init(float fan_in, float fan_out) {
        random_device rd;
        mt19937 gen(rd());
        float std = sqrtf(2.0f / (fan_in + fan_out));
        normal_distribution<float> dist(0.0f, std);
        for (auto& val : data) val = dist(gen);
    }

    void zero_grad() { fill(grad.begin(), grad.end(), 0.0f); }

    float& at(int r, int c) { return data[r * cols + c]; }
    float at(int r, int c) const { return data[r * cols + c]; }
    float& grad_at(int r, int c) { return grad[r * cols + c]; }
};

class RMSNorm {
public:
    fvec weight;
    fvec grad;
    int dim;
    float eps = 1e-6f;

    RMSNorm(int d) : dim(d) {
        weight.resize(d, 1.0f);
        grad.resize(d, 0.0f);
    }

    fvec forward(const fvec& x) {
        fvec out(x.size());
        int seq_len = x.size() / dim;
        for (int s = 0; s < seq_len; ++s) {
            float ss = 0.0f;
            for (int i = 0; i < dim; ++i) {
                float v = x[s * dim + i];
                ss += v * v;
            }
            float scale = 1.0f / sqrtf(ss / dim + eps);
            for (int i = 0; i < dim; ++i) {
                out[s * dim + i] = x[s * dim + i] * scale * weight[i];
            }
        }
        return out;
    }
};

class RotaryEmbedding {
public:
    fvec cos_cache, sin_cache;
    int dim, max_seq;

    RotaryEmbedding(int d, int max_s) : dim(d), max_seq(max_s) {
        cos_cache.resize(max_s * d / 2);
        sin_cache.resize(max_s * d / 2);
        for (int pos = 0; pos < max_s; ++pos) {
            for (int i = 0; i < d / 2; ++i) {
                float freq = 1.0f / powf(10000.0f, 2.0f * i / d);
                float angle = pos * freq;
                cos_cache[pos * (d / 2) + i] = cosf(angle);
                sin_cache[pos * (d / 2) + i] = sinf(angle);
            }
        }
    }

    void apply(fvec& q, fvec& k, int seq_len, int head_dim) {
        int half = head_dim / 2;
        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < half; ++i) {
                float cos_val = cos_cache[s * half + i];
                float sin_val = sin_cache[s * half + i];
                
                float q0 = q[s * head_dim + i];
                float q1 = q[s * head_dim + i + half];
                q[s * head_dim + i] = q0 * cos_val - q1 * sin_val;
                q[s * head_dim + i + half] = q0 * sin_val + q1 * cos_val;
                
                float k0 = k[s * head_dim + i];
                float k1 = k[s * head_dim + i + half];
                k[s * head_dim + i] = k0 * cos_val - k1 * sin_val;
                k[s * head_dim + i + half] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
};

class Attention {
public:
    Tensor wq, wk, wv, wo;
    RMSNorm norm;
    int dim, n_heads, head_dim;
    fvec k_cache, v_cache;
    int cache_len = 0;

    Attention(int d, int nh) : norm(d), dim(d), n_heads(nh), head_dim(d / nh) {
        wq = Tensor(d, d);
        wk = Tensor(d, d);
        wv = Tensor(d, d);
        wo = Tensor(d, d);
        wq.xavier_init(d, d);
        wk.xavier_init(d, d);
        wv.xavier_init(d, d);
        wo.xavier_init(d, d);
    }

    fvec forward(const fvec& x, RotaryEmbedding& rope, bool use_cache = false) {
        int seq_len = x.size() / dim;
        fvec normed = norm.forward(x);
        
        fvec q(seq_len * dim), k(seq_len * dim), v(seq_len * dim);
        
        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < dim; ++i) {
                float qv = 0, kv = 0, vv = 0;
                for (int j = 0; j < dim; ++j) {
                    float inp = normed[s * dim + j];
                    qv += inp * wq.at(j, i);
                    kv += inp * wk.at(j, i);
                    vv += inp * wv.at(j, i);
                }
                q[s * dim + i] = qv;
                k[s * dim + i] = kv;
                v[s * dim + i] = vv;
            }
        }

        for (int h = 0; h < n_heads; ++h) {
            fvec q_head(seq_len * head_dim), k_head(seq_len * head_dim);
            for (int s = 0; s < seq_len; ++s) {
                for (int i = 0; i < head_dim; ++i) {
                    q_head[s * head_dim + i] = q[s * dim + h * head_dim + i];
                    k_head[s * head_dim + i] = k[s * dim + h * head_dim + i];
                }
            }
            rope.apply(q_head, k_head, seq_len, head_dim);
            for (int s = 0; s < seq_len; ++s) {
                for (int i = 0; i < head_dim; ++i) {
                    q[s * dim + h * head_dim + i] = q_head[s * head_dim + i];
                    k[s * dim + h * head_dim + i] = k_head[s * head_dim + i];
                }
            }
        }

        if (use_cache) {
            k_cache.insert(k_cache.end(), k.begin(), k.end());
            v_cache.insert(v_cache.end(), v.begin(), v.end());
            k = k_cache;
            v = v_cache;
            cache_len += seq_len;
        }

        int kv_len = k.size() / dim;
        fvec out(seq_len * dim, 0.0f);
        float scale = 1.0f / sqrtf(head_dim);

        for (int h = 0; h < n_heads; ++h) {
            for (int sq = 0; sq < seq_len; ++sq) {
                fvec scores(kv_len);
                float max_score = -1e9f;
                
                for (int sk = 0; sk < kv_len; ++sk) {
                    if (!use_cache && sk > sq) {
                        scores[sk] = -1e9f;
                        continue;
                    }
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; ++i) {
                        score += q[sq * dim + h * head_dim + i] * k[sk * dim + h * head_dim + i];
                    }
                    scores[sk] = score * scale;
                    max_score = max(max_score, scores[sk]);
                }

                float sum = 0.0f;
                for (int sk = 0; sk < kv_len; ++sk) {
                    scores[sk] = fast_exp(scores[sk] - max_score);
                    sum += scores[sk];
                }
                for (int sk = 0; sk < kv_len; ++sk) scores[sk] /= sum;

                for (int i = 0; i < head_dim; ++i) {
                    float val = 0.0f;
                    for (int sk = 0; sk < kv_len; ++sk) {
                        val += scores[sk] * v[sk * dim + h * head_dim + i];
                    }
                    out[sq * dim + h * head_dim + i] = val;
                }
            }
        }

        fvec proj(seq_len * dim, 0.0f);
        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < dim; ++i) {
                float val = 0.0f;
                for (int j = 0; j < dim; ++j) {
                    val += out[s * dim + j] * wo.at(j, i);
                }
                proj[s * dim + i] = val + x[s * dim + i];
            }
        }
        return proj;
    }

    void clear_cache() {
        k_cache.clear();
        v_cache.clear();
        cache_len = 0;
    }
};

class SwiGLU {
public:
    Tensor w1, w2, w3;
    RMSNorm norm;
    int dim, hidden;

    SwiGLU(int d, int h) : norm(d), dim(d), hidden(h) {
        w1 = Tensor(d, h);
        w2 = Tensor(h, d);
        w3 = Tensor(d, h);
        w1.xavier_init(d, h);
        w2.xavier_init(h, d);
        w3.xavier_init(d, h);
    }

    fvec forward(const fvec& x) {
        int seq_len = x.size() / dim;
        fvec normed = norm.forward(x);
        fvec h1(seq_len * hidden), h3(seq_len * hidden);

        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < hidden; ++i) {
                float v1 = 0, v3 = 0;
                for (int j = 0; j < dim; ++j) {
                    v1 += normed[s * dim + j] * w1.at(j, i);
                    v3 += normed[s * dim + j] * w3.at(j, i);
                }
                h1[s * hidden + i] = silu(v1) * v3;
            }
        }

        fvec out(seq_len * dim);
        for (int s = 0; s < seq_len; ++s) {
            for (int i = 0; i < dim; ++i) {
                float val = 0.0f;
                for (int j = 0; j < hidden; ++j) {
                    val += h1[s * hidden + j] * w2.at(j, i);
                }
                out[s * dim + i] = val + x[s * dim + i];
            }
        }
        return out;
    }
};

class TransformerBlock {
public:
    Attention attn;
    SwiGLU ffn;

    TransformerBlock(int dim, int n_heads, int hidden) 
        : attn(dim, n_heads), ffn(dim, hidden) {}

    fvec forward(const fvec& x, RotaryEmbedding& rope, bool use_cache = false) {
        fvec h = attn.forward(x, rope, use_cache);
        return ffn.forward(h);
    }

    void clear_cache() { attn.clear_cache(); }
};

class Transformer {
public:
    Config cfg;
    Tensor embed, unembed;
    RMSNorm final_norm;
    vector<TransformerBlock> layers;
    RotaryEmbedding rope;
    
    Tensor m_embed, v_embed, m_unembed, v_unembed;
    int step = 0;

    Transformer(const Config& c) 
        : cfg(c), final_norm(c.dim), rope(c.dim / c.n_heads, c.seq_len) {
        embed = Tensor(c.vocab_size, c.dim);
        unembed = Tensor(c.dim, c.vocab_size);
        embed.xavier_init(c.vocab_size, c.dim);
        unembed.xavier_init(c.dim, c.vocab_size);
        
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
        
        for (int s = 0; s < seq_len; ++s) {
            int tok = tokens[s] % cfg.vocab_size;
            for (int i = 0; i < cfg.dim; ++i) {
                x[s * cfg.dim + i] = embed.at(tok, i);
            }
        }

        for (auto& layer : layers) {
            x = layer.forward(x, rope, use_cache);
        }

        x = final_norm.forward(x);

        fvec logits(cfg.vocab_size);
        int last = seq_len - 1;
        for (int i = 0; i < cfg.vocab_size; ++i) {
            float val = 0.0f;
            for (int j = 0; j < cfg.dim; ++j) {
                val += x[last * cfg.dim + j] * unembed.at(j, i);
            }
            logits[i] = val;
        }
        return logits;
    }

    float train_step(const vector<int>& tokens) {
        if (tokens.size() < 2) return 0.0f;
        
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
        
        float loss = -logits[target] + max_logit + logf(sum_exp);
        
        for (int i = 0; i < cfg.vocab_size; ++i) {
            probs[i] = probs[i] / sum_exp - (i == target ? 1.0f : 0.0f);
        }

        ++step;
        float lr = cfg.lr * sqrtf(1.0f - powf(cfg.beta2, step)) / (1.0f - powf(cfg.beta1, step));
        
        for (int i = 0; i < cfg.dim; ++i) {
            for (int j = 0; j < cfg.vocab_size; ++j) {
                float g = probs[j];
                unembed.m[i * cfg.vocab_size + j] = cfg.beta1 * unembed.m[i * cfg.vocab_size + j] + (1 - cfg.beta1) * g;
                unembed.v[i * cfg.vocab_size + j] = cfg.beta2 * unembed.v[i * cfg.vocab_size + j] + (1 - cfg.beta2) * g * g;
                float update = lr * unembed.m[i * cfg.vocab_size + j] / (sqrtf(unembed.v[i * cfg.vocab_size + j]) + cfg.eps);
                unembed.at(i, j) -= update + cfg.weight_decay * lr * unembed.at(i, j);
            }
        }

        int last_tok = input.back() % cfg.vocab_size;
        for (int i = 0; i < cfg.dim; ++i) {
            float g = 0.0f;
            for (int j = 0; j < cfg.vocab_size; ++j) {
                g += probs[j] * unembed.at(i, j);
            }
            embed.m[last_tok * cfg.dim + i] = cfg.beta1 * embed.m[last_tok * cfg.dim + i] + (1 - cfg.beta1) * g;
            embed.v[last_tok * cfg.dim + i] = cfg.beta2 * embed.v[last_tok * cfg.dim + i] + (1 - cfg.beta2) * g * g;
            float update = lr * embed.m[last_tok * cfg.dim + i] / (sqrtf(embed.v[last_tok * cfg.dim + i]) + cfg.eps);
            embed.at(last_tok, i) -= update + cfg.weight_decay * lr * embed.at(last_tok, i);
        }

        return loss;
    }

    void clear_cache() {
        for (auto& layer : layers) layer.clear_cache();
    }

    int sample(const fvec& logits, float temperature = 0.8f, float top_p = 0.9f) {
        fvec probs(logits.size());
        float max_l = *max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = fast_exp((logits[i] - max_l) / temperature);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;

        vector<pair<float, int>> sorted_probs;
        for (size_t i = 0; i < probs.size(); ++i) {
            sorted_probs.emplace_back(probs[i], i);
        }
        sort(sorted_probs.rbegin(), sorted_probs.rend());

        float cumsum = 0.0f;
        int cutoff = sorted_probs.size();
        for (size_t i = 0; i < sorted_probs.size(); ++i) {
            cumsum += sorted_probs[i].first;
            if (cumsum > top_p) {
                cutoff = i + 1;
                break;
            }
        }

        float renorm = 0.0f;
        for (int i = 0; i < cutoff; ++i) renorm += sorted_probs[i].first;
        
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(0.0f, renorm);
        float r = dist(gen);
        
        cumsum = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            cumsum += sorted_probs[i].first;
            if (cumsum >= r) return sorted_probs[i].second;
        }
        return sorted_probs[0].second;
    }
};
