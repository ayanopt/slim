#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>

using namespace std;

using fvec = vector<float>;

inline int num_threads() {
    static int nt = max(1u, thread::hardware_concurrency());
    return nt;
}

inline float fast_exp(float x) {
    x = max(-88.0f, min(88.0f, x));
    return expf(x);
}

inline float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

inline float silu(float x) {
    return x / (1.0f + fast_exp(-x));
}

inline float silu_backward(float x, float grad) {
    float sig = 1.0f / (1.0f + fast_exp(-x));
    return grad * (sig * (1.0f + x * (1.0f - sig)));
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + fast_tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline float gelu_backward(float x, float grad) {
    float c = 0.044715f;
    float s = 0.7978845608f * (x + c * x * x * x);
    float t = fast_tanh(s);
    float dt = 1.0f - t * t;
    float ds = 0.7978845608f * (1.0f + 3.0f * c * x * x);
    return grad * 0.5f * ((1.0f + t) + x * dt * ds);
}

class Tensor {
public:
    fvec data;
    fvec grad;
    fvec m, v;
    int rows, cols;

    Tensor() : rows(0), cols(0) {}
    
    Tensor(int r, int c) : rows(r), cols(c) {
        data.resize(r * c);
        grad.resize(r * c, 0.0f);
        m.resize(r * c, 0.0f);
        v.resize(r * c, 0.0f);
    }

    void he_init() {
        random_device rd;
        mt19937 gen(rd());
        float std = sqrtf(2.0f / rows);
        normal_distribution<float> dist(0.0f, std);
        for (auto& val : data) val = dist(gen);
    }

    void xavier_init() {
        random_device rd;
        mt19937 gen(rd());
        float std = sqrtf(2.0f / (rows + cols));
        normal_distribution<float> dist(0.0f, std);
        for (auto& val : data) val = dist(gen);
    }

    void zero_grad() { 
        fill(grad.begin(), grad.end(), 0.0f); 
    }

    float& at(int r, int c) { return data[r * cols + c]; }
    float at(int r, int c) const { return data[r * cols + c]; }
    float& grad_at(int r, int c) { return grad[r * cols + c]; }
    float grad_at(int r, int c) const { return grad[r * cols + c]; }

    void adamw_update(float lr, float beta1, float beta2, float eps, float wd, int t) {
        float bc1 = 1.0f - powf(beta1, t);
        float bc2 = 1.0f - powf(beta2, t);
        float lr_t = lr * sqrtf(bc2) / bc1;
        
        int n = data.size();
        int nt = num_threads();
        vector<thread> threads;
        
        auto update_chunk = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                float g = max(-1.0f, min(1.0f, grad[i]));
                m[i] = beta1 * m[i] + (1.0f - beta1) * g;
                v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
                float update = lr_t * m[i] / (sqrtf(v[i]) + eps);
                data[i] -= update + wd * lr * data[i];
            }
        };
        
        if (n >= nt * 1000) {
            int chunk = n / nt;
            for (int t = 0; t < nt; ++t) {
                int start = t * chunk;
                int end = (t == nt - 1) ? n : start + chunk;
                threads.emplace_back(update_chunk, start, end);
            }
            for (auto& th : threads) th.join();
        } else {
            update_chunk(0, n);
        }
    }
};

inline void matmul(const fvec& A, const Tensor& B, fvec& C, int M, int K, int N) {
    fill(C.begin(), C.end(), 0.0f);
    int nt = num_threads();
    vector<thread> threads;
    
    auto compute = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int k = 0; k < K; ++k) {
                float a = A[i * K + k];
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a * B.at(k, j);
                }
            }
        }
    };
    
    if (M >= nt * 2) {
        int chunk = M / nt;
        for (int t = 0; t < nt; ++t) {
            int start = t * chunk;
            int end = (t == nt - 1) ? M : start + chunk;
            threads.emplace_back(compute, start, end);
        }
        for (auto& th : threads) th.join();
    } else {
        compute(0, M);
    }
}

inline void matmul_backward_A(fvec& dA, const fvec& dC, const Tensor& B, int M, int K, int N) {
    fill(dA.begin(), dA.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float dc = dC[i * N + j];
            for (int k = 0; k < K; ++k) {
                dA[i * K + k] += dc * B.at(k, j);
            }
        }
    }
}

inline void matmul_backward_B(const fvec& A, const fvec& dC, Tensor& B, int M, int K, int N) {
    int nt = num_threads();
    vector<thread> threads;
    
    auto compute = [&](int start, int end) {
        for (int k = start; k < end; ++k) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < M; ++i) {
                    sum += A[i * K + k] * dC[i * N + j];
                }
                B.grad_at(k, j) += sum;
            }
        }
    };
    
    if (K >= nt * 2) {
        int chunk = K / nt;
        for (int t = 0; t < nt; ++t) {
            int start = t * chunk;
            int end = (t == nt - 1) ? K : start + chunk;
            threads.emplace_back(compute, start, end);
        }
        for (auto& th : threads) th.join();
    } else {
        compute(0, K);
    }
}

inline void softmax(fvec& x, int len) {
    float maxv = *max_element(x.begin(), x.begin() + len);
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        x[i] = fast_exp(x[i] - maxv);
        sum += x[i];
    }
    float inv = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < len; ++i) x[i] *= inv;
}

inline void softmax_backward(const fvec& y, fvec& dy, int len) {
    float dot = 0.0f;
    for (int i = 0; i < len; ++i) dot += y[i] * dy[i];
    for (int i = 0; i < len; ++i) dy[i] = y[i] * (dy[i] - dot);
}
