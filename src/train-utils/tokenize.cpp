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
    return token_indexer.at(tokenize(word));
}

string token_map::get_word(int token) {
    auto it = token_indexer.find(token);
    if (it == token_indexer.end()) {
        cerr << "This word hasn't been tokenized" << endl;
        throw invalid_argument("This token hasn't been observerd " +
                               to_string(token));
    }
    return string_accessor[it->second];
}

void token_map::add(string &word) {
    int token = tokenize(word);
    auto it = token_indexer.try_emplace(token, size());
    if (it.second) {
        string_accessor.push_back(word);
    }
}