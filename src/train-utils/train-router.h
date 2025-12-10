#pragma once
#include "tokenize.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

int train_model(const string& file_path, Tokenizer& tokenizer);
