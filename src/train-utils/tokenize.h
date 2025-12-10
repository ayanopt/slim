#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>

using namespace std;

class Tokenizer {
public:
    static constexpr int PAD_ID = 0;
    static constexpr int UNK_ID = 1;
    static constexpr int BOS_ID = 2;
    static constexpr int EOS_ID = 3;

    Tokenizer() {
        add_token("<PAD>");
        add_token("<UNK>");
        add_token("<BOS>");
        add_token("<EOS>");
    }

    int add_token(const string& token) {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end()) return it->second;
        int id = static_cast<int>(id_to_token.size());
        token_to_id[token] = id;
        id_to_token.push_back(token);
        return id;
    }

    int encode_token(const string& token) const {
        auto it = token_to_id.find(token);
        return it != token_to_id.end() ? it->second : UNK_ID;
    }

    string decode_token(int id) const {
        return (id >= 0 && id < static_cast<int>(id_to_token.size())) ? id_to_token[id] : "<UNK>";
    }

    vector<int> encode(const string& text) {
        vector<int> ids;
        string word;
        for (char c : text) {
            if (isspace(c)) {
                if (!word.empty()) {
                    transform(word.begin(), word.end(), word.begin(), ::tolower);
                    ids.push_back(add_token(word));
                    word.clear();
                }
            } else if (isalnum(c)) {
                word += c;
            }
        }
        if (!word.empty()) {
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            ids.push_back(add_token(word));
        }
        return ids;
    }

    string decode(const vector<int>& ids) const {
        string result;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) result += " ";
            result += decode_token(ids[i]);
        }
        return result;
    }

    size_t vocab_size() const { return id_to_token.size(); }
    const vector<string>& vocabulary() const { return id_to_token; }

private:
    unordered_map<string, int> token_to_id;
    vector<string> id_to_token;
};
