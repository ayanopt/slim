#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include "train-utils/train-router.h"

using namespace std;

vector<string> FLAGS = {"-d", "--directory", "-f", "--file"};

int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "Invalid input!" << endl
         << "Usage: train-model -f \"your_file\"" << endl
         << "Usage: train-model -d \"your_directory\"" << endl;
    return 1;
  }

  string flag = argv[1];

  auto it = find(FLAGS.begin(), FLAGS.end(), argv[1]);
  if (it == FLAGS.end()) {
    cerr << "Invalid flag!" << endl << "Valid flags: " << endl;
    for (string flag : FLAGS) {
      cout << flag << endl;
    }
    return 1;
  }
  if (flag == "-d" || flag == "--directory") {
    string directory = argv[2];
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
      cerr << "Error: Directory does not exist or is not a directory."
           << endl;
      return 1;
    }

    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
      if (std::filesystem::is_regular_file(entry.status())) {
        cout << "Training " << entry.path().filename() << endl;
        return train_model(entry.path());
      } else {
        cerr << "Error: Unable to open " << entry.path().filename() << endl
             << "Remove it and try again" << endl;
        return 1;
      }
    }
  }
  else {
    train_model(argv[2]);
  }
  return 0;
}
