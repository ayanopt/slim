#pragma once
#include "transformer.h"
#include "tokenize.h"
#include <memory>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>

class LLMEngine {
public:
    LLMEngine(const Config& cfg) : config(cfg), model(make_unique<Transformer>(cfg)) {}

    string generate(const string& prompt, int max_tokens = 100, float temp = 0.8f, float top_p = 0.9f) {
        vector<int> tokens = tokenizer.encode(prompt);
        if (tokens.empty()) tokens.push_back(Tokenizer::BOS_ID);
        
        sync_vocab();
        model->clear_cache();
        
        fvec logits = model->forward(tokens, true);
        
        vector<int> generated;
        int consecutive_low_conf = 0;
        
        for (int i = 0; i < max_tokens; ++i) {
            auto [next, confidence] = model->sample_with_confidence(logits, temp, top_p);
            
            if (next == Tokenizer::EOS_ID || next == Tokenizer::PAD_ID) break;
            
            if (confidence < config.min_confidence) {
                consecutive_low_conf++;
                if (consecutive_low_conf >= 3) break;
            } else {
                consecutive_low_conf = 0;
            }
            
            generated.push_back(next);
            logits = model->forward({next}, true);
        }
        
        model->clear_cache();
        return tokenizer.decode(generated);
    }

    void pretrain(const string& corpus_path, int epochs = 3) {
        ifstream f(corpus_path);
        if (!f) {
            cerr << "Cannot read corpus: " << corpus_path << endl;
            return;
        }
        
        stringstream buf;
        buf << f.rdbuf();
        string text = buf.str();
        
        cout << "Building BPE vocabulary from corpus..." << endl;
        tokenizer.train_bpe(text, 8000);
        
        sync_vocab();
        cout << "Vocab size: " << config.vocab_size << endl;
        
        train_on_text(text, epochs, "Pretrain");
    }

    void finetune(const string& text, int epochs = 10) {
        vector<int> tokens = tokenizer.encode(text);
        sync_vocab();
        
        cout << "Finetuning on " << tokens.size() << " tokens for " << epochs << " epochs" << endl;
        train_on_text(text, epochs, "Finetune");
    }

    void train(const string& text, int epochs = 10) {
        if (!tokenizer.is_bpe()) {
            tokenizer.train_bpe(text, 5000);
        }
        
        sync_vocab();
        cout << "Vocab: " << config.vocab_size << endl;
        
        train_on_text(text, epochs, "Train");
    }

    void save(const string& path) {
        ofstream f(path, ios::binary);
        if (!f) { cerr << "Cannot save: " << path << endl; return; }
        
        f.write(reinterpret_cast<const char*>(&config), sizeof(config));
        
        tokenizer.save(f);
        
        auto write_tensor = [&](const Tensor& t) {
            f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
        };
        
        auto write_norm = [&](const RMSNorm& n) {
            f.write(reinterpret_cast<const char*>(n.weight.data()), n.weight.size() * sizeof(float));
        };
        
        write_tensor(model->embed);
        write_tensor(model->unembed);
        write_norm(model->final_norm);
        
        for (auto& layer : model->layers) {
            write_tensor(layer.attn.wq);
            write_tensor(layer.attn.wk);
            write_tensor(layer.attn.wv);
            write_tensor(layer.attn.wo);
            write_norm(layer.attn.norm);
            write_tensor(layer.ffn.w1);
            write_tensor(layer.ffn.w2);
            write_tensor(layer.ffn.w3);
            write_norm(layer.ffn.norm);
        }
        
        int step = model->step;
        f.write(reinterpret_cast<const char*>(&step), sizeof(step));
    }

    void load(const string& path) {
        ifstream f(path, ios::binary);
        if (!f) { cerr << "Cannot load: " << path << endl; return; }
        
        f.read(reinterpret_cast<char*>(&config), sizeof(config));
        
        tokenizer.load(f);
        
        model = make_unique<Transformer>(config);
        
        auto read_tensor = [&](Tensor& t) {
            f.read(reinterpret_cast<char*>(t.data.data()), t.data.size() * sizeof(float));
        };
        
        auto read_norm = [&](RMSNorm& n) {
            f.read(reinterpret_cast<char*>(n.weight.data()), n.weight.size() * sizeof(float));
        };
        
        read_tensor(model->embed);
        read_tensor(model->unembed);
        read_norm(model->final_norm);
        
        for (auto& layer : model->layers) {
            read_tensor(layer.attn.wq);
            read_tensor(layer.attn.wk);
            read_tensor(layer.attn.wv);
            read_tensor(layer.attn.wo);
            read_norm(layer.attn.norm);
            read_tensor(layer.ffn.w1);
            read_tensor(layer.ffn.w2);
            read_tensor(layer.ffn.w3);
            read_norm(layer.ffn.norm);
        }
        
        f.read(reinterpret_cast<char*>(&model->step), sizeof(model->step));
    }

    Config& get_config() { return config; }

private:
    Config config;
    unique_ptr<Transformer> model;
    Tokenizer tokenizer;

    void sync_vocab() {
        int new_size = static_cast<int>(tokenizer.vocab_size());
        if (new_size != config.vocab_size) {
            config.vocab_size = new_size;
            model->resize_vocab(new_size);
        }
    }

    void train_on_text(const string& text, int epochs, const string& phase) {
        vector<int> all_tokens = tokenizer.encode(text);
        if (all_tokens.size() < 2) {
            cerr << "Text too short" << endl;
            return;
        }

        vector<vector<int>> sequences;
        int stride = config.seq_len / 2;
        for (size_t i = 0; i + config.seq_len <= all_tokens.size(); i += stride) {
            vector<int> seq(all_tokens.begin() + i, all_tokens.begin() + i + config.seq_len);
            sequences.push_back(seq);
        }
        if (sequences.empty() && !all_tokens.empty()) {
            sequences.push_back(all_tokens);
        }

        cout << phase << ": " << sequences.size() << " sequences, " << all_tokens.size() << " tokens" << endl;

        auto start_time = chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int count = 0;
            
            for (const auto& seq : sequences) {
                for (size_t i = 2; i <= seq.size(); ++i) {
                    vector<int> sub(seq.begin(), seq.begin() + i);
                    float loss = model->train_step(sub);
                    if (isfinite(loss)) {
                        total_loss += loss;
                        ++count;
                    }
                }
            }
            
            auto now = chrono::high_resolution_clock::now();
            float elapsed = chrono::duration<float>(now - start_time).count();
            
            if (epoch % max(1, epochs / 10) == 0 || epoch == epochs - 1) {
                float avg_loss = count > 0 ? total_loss / count : 0.0f;
                cout << phase << " Epoch " << epoch + 1 << "/" << epochs 
                     << " Loss: " << fixed << setprecision(4) << avg_loss
                     << " Time: " << fixed << setprecision(1) << elapsed << "s" << endl;
            }
        }
    }
};
