#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include "tokenize.h"
using namespace std;

int train_model(string file_path) {
  ifstream file(file_path);
  if (file.bad()) {
    cerr << "Error: Your file path is invalid or the file is potentially corrupted."
         << endl;
    return 1;
  }
  unsigned int n_threads = std::thread::hardware_concurrency();
  cout << n_threads << " thread available" << endl;
  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    while (ss.good()) {      
      string word;
      ss >> word;
      int token = tokenize(word);
      cout << token << endl;
    }
  }
  return 0;
}
