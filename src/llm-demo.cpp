#include "train-utils/llm-engine.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

string read_file(const string& path) {
    ifstream f(path);
    if (!f) { cerr << "Cannot read: " << path << endl; return ""; }
    stringstream buf;
    buf << f.rdbuf();
    return buf.str();
}

void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " <command> [options]\n\n"
         << "Commands:\n"
         << "  pretrain <corpus> <epochs> <model_out>     Train from scratch on large corpus\n"
         << "  finetune <model> <text_file> <epochs> <model_out>  Finetune existing model\n"
         << "  train <text_file> <epochs> <model_out>     Quick train on text file\n"
         << "  generate <model> [prompt] [max_tokens]     Generate text\n"
         << "  chat <model>                               Interactive chat\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    string cmd = argv[1];
    
    Config cfg;
    cfg.dim = 256;
    cfg.n_heads = 8;
    cfg.n_layers = 6;
    cfg.seq_len = 256;
    cfg.hidden_dim = 1024;
    cfg.lr = 1e-4f;
    cfg.weight_decay = 0.1f;
    cfg.min_confidence = 0.15f;

    if (cmd == "pretrain") {
        if (argc < 5) {
            cerr << "Usage: " << argv[0] << " pretrain <corpus> <epochs> <model_out>" << endl;
            return 1;
        }
        
        LLMEngine engine(cfg);
        int epochs = stoi(argv[3]);
        
        engine.pretrain(argv[2], epochs);
        engine.save(argv[4]);
        cout << "Saved: " << argv[4] << endl;
        
    } else if (cmd == "finetune") {
        if (argc < 6) {
            cerr << "Usage: " << argv[0] << " finetune <model> <text_file> <epochs> <model_out>" << endl;
            return 1;
        }
        
        LLMEngine engine(cfg);
        engine.load(argv[2]);
        
        string text = read_file(argv[3]);
        if (text.empty()) return 1;
        
        int epochs = stoi(argv[4]);
        engine.finetune(text, epochs);
        engine.save(argv[5]);
        cout << "Saved: " << argv[5] << endl;
        
    } else if (cmd == "train") {
        if (argc < 5) {
            cerr << "Usage: " << argv[0] << " train <text_file> <epochs> <model_out>" << endl;
            return 1;
        }
        
        string text = read_file(argv[2]);
        if (text.empty()) return 1;
        
        LLMEngine engine(cfg);
        int epochs = stoi(argv[3]);
        
        engine.train(text, epochs);
        engine.save(argv[4]);
        cout << "Saved: " << argv[4] << endl;
        
    } else if (cmd == "generate") {
        if (argc < 3) {
            cerr << "Usage: " << argv[0] << " generate <model> [prompt] [max_tokens]" << endl;
            return 1;
        }
        
        LLMEngine engine(cfg);
        engine.load(argv[2]);
        
        string prompt;
        if (argc > 3) {
            prompt = argv[3];
        } else {
            cout << "Prompt: ";
            getline(cin, prompt);
        }
        
        int max_tokens = argc > 4 ? stoi(argv[4]) : 100;
        
        string output = engine.generate(prompt, max_tokens, 0.8f, 0.9f);
        cout << prompt << " " << output << endl;
        
    } else if (cmd == "chat") {
        if (argc < 3) {
            cerr << "Usage: " << argv[0] << " chat <model>" << endl;
            return 1;
        }
        
        LLMEngine engine(cfg);
        engine.load(argv[2]);
        
        cout << "Chat mode (type 'quit' to exit)" << endl;
        string line;
        while (true) {
            cout << "> ";
            if (!getline(cin, line) || line == "quit") break;
            if (line.empty()) continue;
            
            string response = engine.generate(line, 50, 0.7f, 0.9f);
            cout << response << endl;
        }
        
    } else {
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
