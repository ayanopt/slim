#include <string>
#include <vector>
#include <sstream>
#include <cctype>

using namespace std;

/*
* TODO: revisit tokenizer logic
* A naïve tokenizer. Returns ASCII sum of normalized characters in string
* @param word: string to be tokenized

Example:

cat -> (67 * 100) + (97 * 10) + (116 * 1) = 7786
cAt -> cat -> (67 * 100) + (97 * 10) + (116 * 1) = 7786

*/
int tokenize(string word);