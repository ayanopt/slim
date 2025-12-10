#pragma once
#include "transformer.h"
#include "tokenize.h"
#include <memory>
#include <fstream>
#include <iostream>

class LLMEngine {
public:
    LLMEngine(const Config& cfg) : config(cfg), model(make_unique<Transformer>(cfg)) {}

    string generate(const string& prompt, int max_tokens = 50, float temp = 0.8f, float top_p = 0.9f) {
        vector<int> tokens = tokenizer.encode(prompt);
        if (tokens.empty()) tokens.push_back(Tokenizer::BOS_ID);
        
        sync_vocab();
        model->clear_cache();
        
        fvec logits = model->forward(tokens, true);
        
        vector<int> generated;
        for (int i = 0; i < max_tokens; ++i) {
            int next = model->sample(logits, temp, top_p);
            if (next == Tokenizer::EOS_ID || next == Tokenizer::PAD_ID) break;
            generated.push_back(next);
            logits = model->forward({next}, true);
        }
        
        model->clear_cache();
        return tokenizer.decode(generated);
    }

    void train(const string& text, int epochs = 10) {
        vector<int> all_tokens = tokenizer.encode(text);
        if (all_tokens.size() < 2) {
            cerr << "Text too short" << endl;
            return;
        }
        
        sync_vocab();
        cout << "Vocab: " << config.vocab_size << ", Tokens: " << all_tokens.size() << endl;

        vector<vector<int>> sequences;
        for (size_t i = 0; i + config.seq_len < all_tokens.size(); i += config.seq_len / 2) {
            vector<int> seq(all_tokens.begin() + i, all_tokens.begin() + i + config.seq_len);
            sequences.push_back(seq);
        }
        if (sequences.empty()) {
            sequences.push_back(all_tokens);
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            int count = 0;
            
            for (const auto& seq : sequences) {
                for (size_t i = 2; i <= seq.size(); ++i) {
                    vector<int> sub(seq.begin(), seq.begin() + i);
                    total_loss += model->train_step(sub);
                    ++count;
                }
            }
            
            if (epoch % max(1, epochs / 10) == 0 || epoch == epochs - 1) {
                cout << "Epoch " << epoch << " Loss: " << total_loss / count << endl;
            }
        }
    }

    void save(const string& path) {
        ofstream f(path, ios::binary);
        if (!f) { cerr << "Cannot save: " << path << endl; return; }
        
        f.write(reinterpret_cast<char*>(&config), sizeof(config));
        
        size_t vocab_sz = tokenizer.vocab_size();
        f.write(reinterpret_cast<char*>(&vocab_sz), sizeof(vocab_sz));
        for (const auto& word : tokenizer.vocabulary()) {
            size_t len = word.size();
            f.write(reinterpret_cast<char*>(&len), sizeof(len));
            f.write(word.data(), len);
        }
        
        auto write_tensor = [&](const Tensor& t) {
            f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
        };
        
        write_tensor(model->embed);
        write_tensor(model->unembed);
        
        for (auto& layer : model->layers) {
            write_tensor(layer.attn.wq);
            write_tensor(layer.attn.wk);
            write_tensor(layer.attn.wv);
            write_tensor(layer.attn.wo);
            write_tensor(layer.ffn.w1);
            write_tensor(layer.ffn.w2);
            write_tensor(layer.ffn.w3);
        }
    }

    void load(const string& path) {
        ifstream f(path, ios::binary);
        if (!f) { cerr << "Cannot load: " << path << endl; return; }
        
        f.read(reinterpret_cast<char*>(&config), sizeof(config));
        
        size_t vocab_sz;
        f.read(reinterpret_cast<char*>(&vocab_sz), sizeof(vocab_sz));
        tokenizer = Tokenizer();
        for (size_t i = 4; i < vocab_sz; ++i) {
            size_t len;
            f.read(reinterpret_cast<char*>(&len), sizeof(len));
            string word(len, '\0');
            f.read(&word[0], len);
            tokenizer.add_token(word);
        }
        
        model = make_unique<Transformer>(config);
        
        auto read_tensor = [&](Tensor& t) {
            f.read(reinterpret_cast<char*>(t.data.data()), t.data.size() * sizeof(float));
        };
        
        read_tensor(model->embed);
        read_tensor(model->unembed);
        
        for (auto& layer : model->layers) {
            read_tensor(layer.attn.wq);
            read_tensor(layer.attn.wk);
            read_tensor(layer.attn.wv);
            read_tensor(layer.attn.wo);
            read_tensor(layer.ffn.w1);
            read_tensor(layer.ffn.w2);
            read_tensor(layer.ffn.w3);
        }
    }

    void load_vocab(const string& path) {
        ifstream f(path);
        if (!f) return;
        string word;
        while (getline(f, word)) {
            if (!word.empty()) tokenizer.add_token(word);
        }
    }

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
};
