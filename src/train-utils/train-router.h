#include "tokenize.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
using namespace std;

/*
 * Train model from file_path
 * @param file_path: path to file to train model from
 * @return: 0 if successful, 1 if failed
 */
int train_model(const string &file_path);
