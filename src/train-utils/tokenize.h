#include <cctype>
#include <climits>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/*
* A naïve tokenizer. Returns ASCII sum of normalized characters in string.
* Developed to prevent costly string lookups from the std unordered_map
* @param word: string to be tokenized

Example:

cat -> (67 * 100) + (97 * 10) + (116 * 1) = 7786
cAt -> cat -> (67 * 100) + (97 * 10) + (116 * 1) = 7786

*/
unsigned int tokenize(string &word);

class token_map {
  public:
    token_map();
    unordered_map<int, int> token_indexer;
    vector<string> string_accessor;
    /*
     * Add a word to the token map
     */
    void add(string &word);
    /*
     * Get the index of a word
     */
    int get_index(string &word);
    /*
     * Get the word at a given index
     */
    string get_word(int index);
    /*
     * Get the size of the token map
     */
    size_t size();
};