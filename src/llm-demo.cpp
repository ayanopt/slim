#include "train-utils/llm-engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;

string read_file(const string& path) {
    ifstream f(path);
    if (!f) { cerr << "Cannot read: " << path << endl; return ""; }
    stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <train|generate|chat> [options]\n"
             << "  train <text_file> [epochs] [model_out]\n"
             << "  generate [model] [prompt] [max_tokens]\n"
             << "  chat [model]\n";
        return 1;
    }

    string mode = argv[1];
    
    Config cfg;
    cfg.dim = 128;
    cfg.n_heads = 4;
    cfg.n_layers = 4;
    cfg.seq_len = 128;
    cfg.hidden_dim = 384;
    cfg.dropout = 0.1f;
    cfg.lr = 3e-4f;
    
    LLMEngine engine(cfg);

    if (mode == "train") {
        if (argc < 3) {
            cerr << "Usage: " << argv[0] << " train <text_file> [epochs] [model_out]" << endl;
            return 1;
        }
        
        string text = read_file(argv[2]);
        if (text.empty()) return 1;
        
        int epochs = argc > 3 ? stoi(argv[3]) : 50;
        string model_path = argc > 4 ? argv[4] : "model.bin";
        
        engine.load_vocab("demo/english-10k-words.txt");
        
        auto start = chrono::high_resolution_clock::now();
        engine.train(text, epochs);
        auto end = chrono::high_resolution_clock::now();
        
        cout << "Training time: " 
             << chrono::duration_cast<chrono::seconds>(end - start).count() << "s" << endl;
        
        engine.save(model_path);
        cout << "Saved: " << model_path << endl;
        
    } else if (mode == "generate") {
        string model_path = argc > 2 ? argv[2] : "model.bin";
        engine.load(model_path);
        
        string prompt;
        if (argc > 3) {
            prompt = argv[3];
        } else {
            cout << "Prompt: ";
            getline(cin, prompt);
        }
        
        int max_tokens = argc > 4 ? stoi(argv[4]) : 50;
        
        cout << "Generating..." << endl;
        string output = engine.generate(prompt, max_tokens, 0.8f, 0.9f);
        cout << prompt << " " << output << endl;
        
    } else if (mode == "chat") {
        string model_path = argc > 2 ? argv[2] : "model.bin";
        engine.load(model_path);
        
        cout << "Chat mode (type 'quit' to exit)" << endl;
        string line;
        while (true) {
            cout << "> ";
            if (!getline(cin, line) || line == "quit") break;
            if (line.empty()) continue;
            
            string response = engine.generate(line, 30, 0.9f, 0.95f);
            cout << response << endl;
        }
        
    } else {
        cerr << "Unknown mode: " << mode << endl;
        return 1;
    }

    return 0;
}
