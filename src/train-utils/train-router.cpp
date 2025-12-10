#include "train-router.h"

int train_model(const string& file_path, Tokenizer& tokenizer) {
    ifstream file(file_path);
    if (!file) {
        cerr << "Error: Invalid file path" << endl;
        return 1;
    }
    string line;
    while (getline(file, line)) {
        tokenizer.encode(line);
    }
    return 0;
}
