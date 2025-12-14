#pragma once
#include "tensor.h"

struct LayerCache {
    fvec input;
    fvec normed;
    fvec q, k, v;
    fvec attn_scores;
    fvec attn_out;
    fvec h1, h3;
    fvec ffn_out;
};

class RMSNorm {
public:
    fvec weight;
    fvec weight_grad;
    fvec m, v;
    fvec scales;
    int dim;
    float eps = 1e-6f;

    RMSNorm(int d) : dim(d) {
        weight.resize(d, 1.0f);
        weight_grad.resize(d, 0.0f);
        m.resize(d, 0.0f);
        v.resize(d, 0.0f);
    }

    fvec forward(const fvec& x, fvec& scales_out) {
        int seq_len = x.size() / dim;
        fvec out(x.size());
        scales_out.resize(seq_len);
        
        for (int s = 0; s < seq_len; ++s) {
            float ss = 0.0f;
            for (int i = 0; i < dim; ++i) {
                float val = x[s * dim + i];
                ss += val * val;
            }
            float scale = 1.0f / sqrtf(ss / dim + eps);
            scales_out[s] = scale;
            for (int i = 0; i < dim; ++i) {
                out[s * dim + i] = x[s * dim + i] * scale * weight[i];
            }
        }
        return out;
    }

    void backward(const fvec& x, const fvec& grad_out, const fvec& scales_in, fvec& grad_in) {
        int seq_len = x.size() / dim;
        grad_in.resize(x.size());
        
        for (int s = 0; s < seq_len; ++s) {
            float scale = scales_in[s];
            float sum_grad_x = 0.0f;
            
            for (int i = 0; i < dim; ++i) {
                weight_grad[i] += grad_out[s * dim + i] * x[s * dim + i] * scale;
                sum_grad_x += grad_out[s * dim + i] * weight[i] * x[s * dim + i];
            }
            
            float coeff = scale * scale * scale / dim;
            for (int i = 0; i < dim; ++i) {
                grad_in[s * dim + i] = grad_out[s * dim + i] * weight[i] * scale 
                                      - coeff * sum_grad_x * x[s * dim + i];
            }
        }
    }

    void update(float lr, float beta1, float beta2, float eps_adam, float wd, int t) {
        float bc1 = 1.0f - powf(beta1, t);
        float bc2 = 1.0f - powf(beta2, t);
        float lr_t = lr * sqrtf(bc2) / bc1;
        
        for (int i = 0; i < dim; ++i) {
            float g = max(-1.0f, min(1.0f, weight_grad[i]));
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
            weight[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps_adam) + wd * lr * weight[i];
            weight_grad[i] = 0.0f;
        }
    }
};

class RotaryEmbedding {
public:
    fvec cos_cache, sin_cache;
    int dim, max_seq;

    RotaryEmbedding(int d, int max_s) : dim(d), max_seq(max_s) {
        cos_cache.resize(max_s * d / 2);
        sin_cache.resize(max_s * d / 2);
        float inv_freq_base = 1.0f / 10000.0f;
        for (int pos = 0; pos < max_s; ++pos) {
            for (int i = 0; i < d / 2; ++i) {
                float freq = powf(inv_freq_base, 2.0f * i / d);
                float angle = pos * freq;
                cos_cache[pos * (d / 2) + i] = cosf(angle);
                sin_cache[pos * (d / 2) + i] = sinf(angle);
            }
        }
    }

    void apply(fvec& q, fvec& k, int seq_len, int n_heads, int head_dim, int offset = 0) {
        int half = head_dim / 2;
        int nt = num_threads();
        vector<thread> threads;
        
        auto apply_rope = [&](int h_start, int h_end) {
            for (int h = h_start; h < h_end; ++h) {
                for (int s = 0; s < seq_len; ++s) {
                    int pos = s + offset;
                    for (int i = 0; i < half; ++i) {
                        float cos_val = cos_cache[pos * half + i];
                        float sin_val = sin_cache[pos * half + i];
                        
                        int idx = s * n_heads * head_dim + h * head_dim;
                        float q0 = q[idx + i], q1 = q[idx + i + half];
                        q[idx + i] = q0 * cos_val - q1 * sin_val;
                        q[idx + i + half] = q0 * sin_val + q1 * cos_val;
                        
                        float k0 = k[idx + i], k1 = k[idx + i + half];
                        k[idx + i] = k0 * cos_val - k1 * sin_val;
                        k[idx + i + half] = k0 * sin_val + k1 * cos_val;
                    }
                }
            }
        };
        
        if (n_heads >= nt) {
            int chunk = n_heads / nt;
            for (int t = 0; t < nt; ++t) {
                int start = t * chunk;
                int end = (t == nt - 1) ? n_heads : start + chunk;
                threads.emplace_back(apply_rope, start, end);
            }
            for (auto& th : threads) th.join();
        } else {
            apply_rope(0, n_heads);
        }
    }

    void apply_backward(fvec& dq, fvec& dk, int seq_len, int n_heads, int head_dim, int offset = 0) {
        int half = head_dim / 2;
        for (int h = 0; h < n_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int pos = s + offset;
                for (int i = 0; i < half; ++i) {
                    float cos_val = cos_cache[pos * half + i];
                    float sin_val = sin_cache[pos * half + i];
                    
                    int idx = s * n_heads * head_dim + h * head_dim;
                    float dq0 = dq[idx + i], dq1 = dq[idx + i + half];
                    dq[idx + i] = dq0 * cos_val + dq1 * sin_val;
                    dq[idx + i + half] = -dq0 * sin_val + dq1 * cos_val;
                    
                    float dk0 = dk[idx + i], dk1 = dk[idx + i + half];
                    dk[idx + i] = dk0 * cos_val + dk1 * sin_val;
                    dk[idx + i + half] = -dk0 * sin_val + dk1 * cos_val;
                }
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
        wq = Tensor(d, d); wq.xavier_init();
        wk = Tensor(d, d); wk.xavier_init();
        wv = Tensor(d, d); wv.xavier_init();
        wo = Tensor(d, d); wo.xavier_init();
    }

    fvec forward(const fvec& x, RotaryEmbedding& rope, LayerCache& cache, bool use_kv_cache = false) {
        int seq_len = x.size() / dim;
        cache.input = x;
        
        fvec scales;
        cache.normed = norm.forward(x, scales);
        cache.attn_scores = scales;
        
        cache.q.resize(seq_len * dim);
        cache.k.resize(seq_len * dim);
        cache.v.resize(seq_len * dim);
        
        matmul(cache.normed, wq, cache.q, seq_len, dim, dim);
        matmul(cache.normed, wk, cache.k, seq_len, dim, dim);
        matmul(cache.normed, wv, cache.v, seq_len, dim, dim);
        
        fvec q_rope = cache.q, k_rope = cache.k;
        rope.apply(q_rope, k_rope, seq_len, n_heads, head_dim, use_kv_cache ? cache_len : 0);
        
        if (use_kv_cache) {
            k_cache.insert(k_cache.end(), k_rope.begin(), k_rope.end());
            v_cache.insert(v_cache.end(), cache.v.begin(), cache.v.end());
            cache_len += seq_len;
        }
        
        const fvec& k_use = use_kv_cache ? k_cache : k_rope;
        const fvec& v_use = use_kv_cache ? v_cache : cache.v;
        int kv_len = k_use.size() / dim;
        
        cache.attn_out.resize(seq_len * dim);
        float scale = 1.0f / sqrtf(head_dim);
        
        vector<thread> threads;
        
        auto compute_head = [&](int h) {
            for (int sq = 0; sq < seq_len; ++sq) {
                fvec scores(kv_len);
                
                for (int sk = 0; sk < kv_len; ++sk) {
                    if (!use_kv_cache && sk > sq) {
                        scores[sk] = -1e9f;
                    } else {
                        float dot = 0.0f;
                        for (int i = 0; i < head_dim; ++i) {
                            dot += q_rope[sq * dim + h * head_dim + i] * k_use[sk * dim + h * head_dim + i];
                        }
                        scores[sk] = dot * scale;
                    }
                }
                
                softmax(scores, kv_len);
                
                for (int i = 0; i < head_dim; ++i) {
                    float val = 0.0f;
                    for (int sk = 0; sk < kv_len; ++sk) {
                        val += scores[sk] * v_use[sk * dim + h * head_dim + i];
                    }
                    cache.attn_out[sq * dim + h * head_dim + i] = val;
                }
            }
        };
        
        if (n_heads >= 2) {
            for (int h = 0; h < n_heads; ++h) threads.emplace_back(compute_head, h);
            for (auto& th : threads) th.join();
        } else {
            for (int h = 0; h < n_heads; ++h) compute_head(h);
        }
        
        fvec out(seq_len * dim);
        matmul(cache.attn_out, wo, out, seq_len, dim, dim);
        
        for (int i = 0; i < seq_len * dim; ++i) out[i] += x[i];
        
        return out;
    }

    void backward(const fvec& grad_out, RotaryEmbedding& rope, LayerCache& cache, fvec& grad_in) {
        int seq_len = cache.input.size() / dim;
        
        fvec grad_residual = grad_out;
        
        fvec grad_attn_out(seq_len * dim);
        matmul_backward_A(grad_attn_out, grad_out, wo, seq_len, dim, dim);
        matmul_backward_B(cache.attn_out, grad_out, wo, seq_len, dim, dim);
        
        fvec grad_q(seq_len * dim, 0.0f);
        fvec grad_k(seq_len * dim, 0.0f);
        fvec grad_v(seq_len * dim, 0.0f);
        
        float scale = 1.0f / sqrtf(head_dim);
        
        for (int h = 0; h < n_heads; ++h) {
            for (int sq = 0; sq < seq_len; ++sq) {
                fvec scores(seq_len);
                for (int sk = 0; sk <= sq; ++sk) {
                    float dot = 0.0f;
                    for (int i = 0; i < head_dim; ++i) {
                        dot += cache.q[sq * dim + h * head_dim + i] * cache.k[sk * dim + h * head_dim + i];
                    }
                    scores[sk] = dot * scale;
                }
                for (int sk = sq + 1; sk < seq_len; ++sk) scores[sk] = -1e9f;
                
                softmax(scores, seq_len);
                
                fvec grad_scores(seq_len, 0.0f);
                for (int sk = 0; sk < seq_len; ++sk) {
                    for (int i = 0; i < head_dim; ++i) {
                        grad_scores[sk] += grad_attn_out[sq * dim + h * head_dim + i] * cache.v[sk * dim + h * head_dim + i];
                        grad_v[sk * dim + h * head_dim + i] += scores[sk] * grad_attn_out[sq * dim + h * head_dim + i];
                    }
                }
                
                softmax_backward(scores, grad_scores, seq_len);
                
                for (int sk = 0; sk <= sq; ++sk) {
                    float gs = grad_scores[sk] * scale;
                    for (int i = 0; i < head_dim; ++i) {
                        grad_q[sq * dim + h * head_dim + i] += gs * cache.k[sk * dim + h * head_dim + i];
                        grad_k[sk * dim + h * head_dim + i] += gs * cache.q[sq * dim + h * head_dim + i];
                    }
                }
            }
        }
        
        rope.apply_backward(grad_q, grad_k, seq_len, n_heads, head_dim);
        
        fvec grad_normed(seq_len * dim, 0.0f);
        matmul_backward_A(grad_normed, grad_q, wq, seq_len, dim, dim);
        matmul_backward_B(cache.normed, grad_q, wq, seq_len, dim, dim);
        
        fvec temp(seq_len * dim);
        matmul_backward_A(temp, grad_k, wk, seq_len, dim, dim);
        for (int i = 0; i < seq_len * dim; ++i) grad_normed[i] += temp[i];
        matmul_backward_B(cache.normed, grad_k, wk, seq_len, dim, dim);
        
        matmul_backward_A(temp, grad_v, wv, seq_len, dim, dim);
        for (int i = 0; i < seq_len * dim; ++i) grad_normed[i] += temp[i];
        matmul_backward_B(cache.normed, grad_v, wv, seq_len, dim, dim);
        
        norm.backward(cache.input, grad_normed, cache.attn_scores, grad_in);
        
        for (int i = 0; i < seq_len * dim; ++i) grad_in[i] += grad_residual[i];
    }

    void update(float lr, float beta1, float beta2, float eps, float wd, int t) {
        wq.adamw_update(lr, beta1, beta2, eps, wd, t);
        wk.adamw_update(lr, beta1, beta2, eps, wd, t);
        wv.adamw_update(lr, beta1, beta2, eps, wd, t);
        wo.adamw_update(lr, beta1, beta2, eps, wd, t);
        norm.update(lr, beta1, beta2, eps, wd, t);
    }

    void zero_grad() {
        wq.zero_grad(); wk.zero_grad(); wv.zero_grad(); wo.zero_grad();
    }

    void clear_cache() {
        k_cache.clear(); v_cache.clear(); cache_len = 0;
    }
};

class FeedForward {
public:
    Tensor w1, w2, w3;
    RMSNorm norm;
    int dim, hidden;

    FeedForward(int d, int h) : norm(d), dim(d), hidden(h) {
        w1 = Tensor(d, h); w1.he_init();
        w2 = Tensor(h, d); w2.xavier_init();
        w3 = Tensor(d, h); w3.he_init();
    }

    fvec forward(const fvec& x, LayerCache& cache) {
        int seq_len = x.size() / dim;
        cache.input = x;
        
        fvec scales;
        cache.normed = norm.forward(x, scales);
        cache.attn_scores = scales;
        
        cache.h1.resize(seq_len * hidden);
        cache.h3.resize(seq_len * hidden);
        
        matmul(cache.normed, w1, cache.h1, seq_len, dim, hidden);
        matmul(cache.normed, w3, cache.h3, seq_len, dim, hidden);
        
        cache.ffn_out.resize(seq_len * hidden);
        for (int i = 0; i < seq_len * hidden; ++i) {
            cache.ffn_out[i] = silu(cache.h1[i]) * cache.h3[i];
        }
        
        fvec out(seq_len * dim);
        matmul(cache.ffn_out, w2, out, seq_len, hidden, dim);
        
        for (int i = 0; i < seq_len * dim; ++i) out[i] += x[i];
        
        return out;
    }

    void backward(const fvec& grad_out, LayerCache& cache, fvec& grad_in) {
        int seq_len = cache.input.size() / dim;
        
        fvec grad_residual = grad_out;
        
        fvec grad_ffn_out(seq_len * hidden);
        matmul_backward_A(grad_ffn_out, grad_out, w2, seq_len, hidden, dim);
        matmul_backward_B(cache.ffn_out, grad_out, w2, seq_len, hidden, dim);
        
        fvec grad_h1(seq_len * hidden);
        fvec grad_h3(seq_len * hidden);
        
        for (int i = 0; i < seq_len * hidden; ++i) {
            grad_h1[i] = silu_backward(cache.h1[i], grad_ffn_out[i] * cache.h3[i]);
            grad_h3[i] = grad_ffn_out[i] * silu(cache.h1[i]);
        }
        
        fvec grad_normed(seq_len * dim, 0.0f);
        matmul_backward_A(grad_normed, grad_h1, w1, seq_len, dim, hidden);
        matmul_backward_B(cache.normed, grad_h1, w1, seq_len, dim, hidden);
        
        fvec temp(seq_len * dim);
        matmul_backward_A(temp, grad_h3, w3, seq_len, dim, hidden);
        for (int i = 0; i < seq_len * dim; ++i) grad_normed[i] += temp[i];
        matmul_backward_B(cache.normed, grad_h3, w3, seq_len, dim, hidden);
        
        norm.backward(cache.input, grad_normed, cache.attn_scores, grad_in);
        
        for (int i = 0; i < seq_len * dim; ++i) grad_in[i] += grad_residual[i];
    }

    void update(float lr, float beta1, float beta2, float eps, float wd, int t) {
        w1.adamw_update(lr, beta1, beta2, eps, wd, t);
        w2.adamw_update(lr, beta1, beta2, eps, wd, t);
        w3.adamw_update(lr, beta1, beta2, eps, wd, t);
        norm.update(lr, beta1, beta2, eps, wd, t);
    }

    void zero_grad() {
        w1.zero_grad(); w2.zero_grad(); w3.zero_grad();
    }
};

class TransformerBlock {
public:
    Attention attn;
    FeedForward ffn;
    LayerCache attn_cache, ffn_cache;

    TransformerBlock(int dim, int n_heads, int hidden) 
        : attn(dim, n_heads), ffn(dim, hidden) {}

    fvec forward(const fvec& x, RotaryEmbedding& rope, bool use_cache = false) {
        fvec h = attn.forward(x, rope, attn_cache, use_cache);
        return ffn.forward(h, ffn_cache);
    }

    void backward(const fvec& grad_out, RotaryEmbedding& rope, fvec& grad_in) {
        fvec grad_ffn_in;
        ffn.backward(grad_out, ffn_cache, grad_ffn_in);
        attn.backward(grad_ffn_in, rope, attn_cache, grad_in);
    }

    void update(float lr, float beta1, float beta2, float eps, float wd, int t) {
        attn.update(lr, beta1, beta2, eps, wd, t);
        ffn.update(lr, beta1, beta2, eps, wd, t);
    }

    void zero_grad() {
        attn.zero_grad();
        ffn.zero_grad();
    }

    void clear_cache() { attn.clear_cache(); }
};
