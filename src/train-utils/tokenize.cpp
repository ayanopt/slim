#include <string>
#include <vector>
#include <sstream>
#include <cctype>

using namespace std;

int tokenize(string word) {
    int curr_sum = 0;
    for (char &c : word) {
        curr_sum += tolower(c);
    }
    return curr_sum;
}