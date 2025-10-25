#include "llm-engine.h"
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>

LLMEngine::LLMEngine(const TransformerConfig& config) : config(config) {
    model = make_unique<Transformer>(config);
}

vector<int> LLMEngine::tokenize_text(const string& text) {
    vector<int> tokens;
    stringstream ss(text);
    string word;
    
    while (ss >> word) {
        transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(remove_if(word.begin(), word.end(), [](char c) {
            return !isalnum(c);
        }), word.end());
        
        if (!word.empty()) {
            tokenizer.add(word);
            tokens.push_back(tokenizer.get_index(word));
        }
    }
    
    return tokens;
}

string LLMEngine::detokenize(const vector<int>& tokens) {
    string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += " ";
        try {
            result += tokenizer.get_word(tokens[i]);
        } catch (const invalid_argument&) {
            result += "<UNK>";
        }
    }
    return result;
}

vector<vector<int>> LLMEngine::create_batches(const vector<int>& tokens, int batch_size) {
    vector<vector<int>> batches;
    
    for (size_t i = 0; i < tokens.size(); i += batch_size) {
        size_t end = min(i + batch_size, tokens.size());
        batches.emplace_back(tokens.begin() + i, tokens.begin() + end);
    }
    
    return batches;
}

int LLMEngine::sample_next_token(const fvector& logits, float temperature) {
    vector<float> probs = logits;
    
    for (auto& prob : probs) {
        prob /= temperature;
    }
    
    float max_logit = *max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    
    for (auto& prob : probs) {
        prob = exp(prob - max_logit);
        sum += prob;
    }
    
    for (auto& prob : probs) {
        prob /= sum;
    }
    
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<int> dist(probs.begin(), probs.end());
    
    return dist(gen);
}

string LLMEngine::generate(const string& prompt, int max_tokens) {
    vector<int> input_tokens = tokenize_text(prompt);
    
    if (input_tokens.empty()) {
        return "";
    }
    
    vector<int> generated_tokens = input_tokens;
    
    for (int i = 0; i < max_tokens; ++i) {
        if (generated_tokens.size() >= static_cast<size_t>(config.seq_len)) {
            generated_tokens.erase(generated_tokens.begin());
        }
        
        fvector logits = model->forward(generated_tokens);
        int next_token = sample_next_token(logits, 0.8f);
        
        generated_tokens.push_back(next_token);
        
        if (next_token == 0) break;
    }
    
    vector<int> output_tokens(generated_tokens.begin() + input_tokens.size(), 
                             generated_tokens.end());
    
    return detokenize(output_tokens);
}

void LLMEngine::train_on_text(const string& text_data, int epochs) {
    vector<int> tokens = tokenize_text(text_data);
    
    if (tokens.size() < static_cast<size_t>(config.seq_len)) {
        cerr << "Text too short for training" << endl;
        return;
    }
    
    config.vocab_size = tokenizer.size();
    model = make_unique<Transformer>(config);
    
    vector<vector<int>> batches = create_batches(tokens, config.seq_len);
    model->train(batches, epochs);
}

void LLMEngine::save_model(const string& filepath) {
    ofstream file(filepath, ios::binary);
    if (!file) {
        cerr << "Cannot save model to " << filepath << endl;
        return;
    }
    
    file.write(reinterpret_cast<const char*>(&config), sizeof(config));
    
    size_t vocab_size = tokenizer.size();
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    
    for (const auto& word : tokenizer.string_accessor) {
        size_t len = word.length();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(word.c_str(), len);
    }
    
    file.close();
}

void LLMEngine::load_model(const string& filepath) {
    ifstream file(filepath, ios::binary);
    if (!file) {
        cerr << "Cannot load model from " << filepath << endl;
        return;
    }
    
    file.read(reinterpret_cast<char*>(&config), sizeof(config));
    
    size_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    
    tokenizer = token_map();
    
    for (size_t i = 0; i < vocab_size; ++i) {
        size_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        
        string word(len, '\0');
        file.read(&word[0], len);
        
        tokenizer.add(word);
    }
    
    model = make_unique<Transformer>(config);
    file.close();
}