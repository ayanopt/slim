#include "tokenize.h"

int tokenize(string &word) {
    unsigned long int curr_sum = 0;
    for (size_t i = 0; i < word.size(); i++) {
        curr_sum += pow(10, (word.size() - i - 1)) * tolower(word[i]);
    }
    if (curr_sum == ULONG_MAX) {
        cerr << "Overflow at word: " + word << endl;
    }
    return curr_sum % INT32_MAX;
}