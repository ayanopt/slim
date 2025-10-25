#pragma once
#include "transformer.h"
#include "tokenize.h"
#include <memory>

class LLMEngine {
public:
    LLMEngine(const TransformerConfig& config);
    
    /**
     * Generate text from input prompt
     * @param prompt Input text string
     * @param max_tokens Maximum tokens to generate
     * @return Generated text
     */
    string generate(const string& prompt, int max_tokens = 50);
    
    /**
     * Train the model on text data
     * @param text_data Training text
     * @param epochs Number of training epochs
     */
    void train_on_text(const string& text_data, int epochs = 10);
    
    /**
     * Save model weights to file
     * @param filepath Path to save model
     */
    void save_model(const string& filepath);
    
    /**
     * Load model weights from file
     * @param filepath Path to load model from
     */
    void load_model(const string& filepath);

private:
    unique_ptr<Transformer> model;
    token_map tokenizer;
    TransformerConfig config;
    
    vector<int> tokenize_text(const string& text);
    string detokenize(const vector<int>& tokens);
    vector<vector<int>> create_batches(const vector<int>& tokens, int batch_size);
    int sample_next_token(const fvector& logits, float temperature = 1.0f);
};