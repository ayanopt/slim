#include "tokenize.h"

unsigned int tokenize(string &word) {
    unsigned long int curr_sum = 0;
    for (size_t i = 0; i < word.size(); i++) {
        curr_sum += pow(10, (word.size() - i - 1)) * tolower(word[i]);
    }
    if (curr_sum == ULONG_MAX) {
        cerr << "Overflow at word: " + word << endl;
    }
    return curr_sum % INT32_MAX;
}

size_t token_map::size() { return string_accessor.size(); }

int token_map::get_index(string &word) {
    int token = tokenize(word);
    auto it = token_indexer.find(token);
    if (it == token_indexer.end()) {
        add(word);
        return token_indexer.at(token);
    }
    return it->second;
}

string token_map::get_word(int index) {
    if (index >= 0 && index < static_cast<int>(string_accessor.size())) {
        return string_accessor[index];
    }
    throw invalid_argument("Invalid token index: " + to_string(index));
}

void token_map::add(string &word) {
    int token = tokenize(word);
    auto it = token_indexer.try_emplace(token, size());
    if (it.second) {
        string_accessor.push_back(word);
    }
}