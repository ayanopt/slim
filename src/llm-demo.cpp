#include "train-utils/llm-engine.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

string read_file(const string& filepath) {
    ifstream file(filepath);
    if (!file) {
        cerr << "Cannot read file: " << filepath << endl;
        return "";
    }
    
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <train|generate> [options]" << endl;
        return 1;
    }
    
    string mode = argv[1];
    
    TransformerConfig config;
    config.d_model = 128;
    config.n_heads = 8;
    config.n_layers = 6;
    config.seq_len = 64;
    config.d_ff = 512;
    config.dropout = 0.1f;
    config.vocab_size = 10000;
    
    LLMEngine engine(config);
    
    if (mode == "train") {
        if (argc < 3) {
            cerr << "Usage: " << argv[0] << " train <text_file>" << endl;
            return 1;
        }
        
        string text_data = read_file(argv[2]);
        if (text_data.empty()) {
            return 1;
        }
        
        cout << "Training on " << text_data.length() << " characters..." << endl;
        engine.train_on_text(text_data, 5);
        
        engine.save_model("model.bin");
        cout << "Model saved to model.bin" << endl;
        
    } else if (mode == "generate") {
        engine.load_model("model.bin");
        
        string prompt;
        if (argc >= 3) {
            prompt = argv[2];
        } else {
            cout << "Enter prompt: ";
            getline(cin, prompt);
        }
        
        cout << "Generating text..." << endl;
        string generated = engine.generate(prompt, 30);
        cout << "Generated: " << generated << endl;
        
    } else {
        cerr << "Unknown mode: " << mode << endl;
        return 1;
    }
    
    return 0;
}