#include "train-router.h"

int train_model(const string &file_path) {
    ifstream file(file_path);
    if (file.bad()) {
        cerr << "Error: Your file path is invalid or the file is potentially "
             << "corrupted." << endl;
        return 1;
    }
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        while (ss.good()) {
            string word;
            ss >> word;
            int token = tokenize(word);
        }
    }
    return 0;
}
