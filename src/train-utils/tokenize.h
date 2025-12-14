#pragma once
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

/*
My very own Tokenizer class, which is a wrapper around a vector of tokens and a
map from tokens to ids. It also has a method to train a BPE tokenizer on a given
text.
*/
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

    int add_token(const string &token) {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end())
            return it->second;
        int id = static_cast<int>(id_to_token.size());
        token_to_id[token] = id;
        id_to_token.push_back(token);
        return id;
    }

    int get_token(const string &token) const {
        auto it = token_to_id.find(token);
        return it != token_to_id.end() ? it->second : UNK_ID;
    }

    string get_word(int id) const {
        return (id >= 0 && id < static_cast<int>(id_to_token.size()))
                   ? id_to_token[id]
                   : "<UNK>";
    }

    void train_bpe(const string &text, int num_merges = 5000) {
        unordered_map<string, int> word_freq;
        string word;

        for (char c : text) {
            if (isspace(c)) {
                if (!word.empty()) {
                    transform(word.begin(), word.end(), word.begin(),
                              ::tolower);
                    word_freq[word]++;
                    word.clear();
                }
            } else if (isalnum(c) || c == '\'') {
                word += c;
            }
        }
        if (!word.empty()) {
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            word_freq[word]++;
        }

        unordered_map<string, vector<string>> word_splits;
        for (auto &[w, _] : word_freq) {
            vector<string> chars;
            for (char c : w)
                chars.push_back(string(1, c));
            if (!chars.empty())
                chars.back() += "</w>";
            word_splits[w] = chars;
            for (auto &ch : chars)
                add_token(ch);
        }

        for (int merge = 0; merge < num_merges; ++merge) {
            unordered_map<string, int> pair_freq;
            for (auto &[w, freq] : word_freq) {
                auto &splits = word_splits[w];
                for (size_t i = 0; i + 1 < splits.size(); ++i) {
                    pair_freq[splits[i] + "\t" + splits[i + 1]] += freq;
                }
            }

            if (pair_freq.empty())
                break;

            string best_pair;
            int best_freq = 0;
            for (auto &[p, f] : pair_freq) {
                if (f > best_freq) {
                    best_freq = f;
                    best_pair = p;
                }
            }

            size_t sp = best_pair.find('\t');
            string first = best_pair.substr(0, sp);
            string second = best_pair.substr(sp + 1);
            string merged = first + second;
            add_token(merged);
            merges.push_back({first, second});

            for (auto &[w, _] : word_freq) {
                auto &splits = word_splits[w];
                vector<string> new_splits;
                for (size_t i = 0; i < splits.size(); ++i) {
                    if (i + 1 < splits.size() && splits[i] == first &&
                        splits[i + 1] == second) {
                        new_splits.push_back(merged);
                        ++i;
                    } else {
                        new_splits.push_back(splits[i]);
                    }
                }
                splits = new_splits;
            }
        }
        use_bpe = true;
    }

    vector<int> encode(const string &text) {
        vector<int> ids;
        string word;

        for (char c : text) {
            if (isspace(c)) {
                if (!word.empty()) {
                    transform(word.begin(), word.end(), word.begin(),
                              ::tolower);
                    encode_word(word, ids);
                    word.clear();
                }
            } else if (isalnum(c) || c == '\'') {
                word += c;
            }
        }
        if (!word.empty()) {
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            encode_word(word, ids);
        }
        return ids;
    }

    string decode(const vector<int> &ids) const {
        string result;
        for (size_t i = 0; i < ids.size(); ++i) {
            string tok = get_word(ids[i]);
            if (tok[0] == '<')
                continue;

            if (tok.size() >= 4 && tok.substr(tok.size() - 4) == "</w>") {
                result += tok.substr(0, tok.size() - 4);
                if (i + 1 < ids.size())
                    result += " ";
            } else if (use_bpe) {
                result += tok;
            } else {
                if (!result.empty() && result.back() != ' ')
                    result += " ";
                result += tok;
            }
        }
        return result;
    }

    size_t vocab_size() const { return id_to_token.size(); }
    const vector<string> &vocabulary() const { return id_to_token; }
    const vector<pair<string, string>> &get_merges() const { return merges; }
    bool is_bpe() const { return use_bpe; }

    void save(ostream &out) const {
        size_t vs = id_to_token.size();
        out.write(reinterpret_cast<const char *>(&vs), sizeof(vs));
        for (const auto &tok : id_to_token) {
            size_t len = tok.size();
            out.write(reinterpret_cast<const char *>(&len), sizeof(len));
            out.write(tok.data(), len);
        }
        size_t ms = merges.size();
        out.write(reinterpret_cast<const char *>(&ms), sizeof(ms));
        for (const auto &[f, s] : merges) {
            size_t fl = f.size(), sl = s.size();
            out.write(reinterpret_cast<const char *>(&fl), sizeof(fl));
            out.write(f.data(), fl);
            out.write(reinterpret_cast<const char *>(&sl), sizeof(sl));
            out.write(s.data(), sl);
        }
        out.write(reinterpret_cast<const char *>(&use_bpe), sizeof(use_bpe));
    }

    void load(istream &in) {
        token_to_id.clear();
        id_to_token.clear();
        merges.clear();

        size_t vs;
        in.read(reinterpret_cast<char *>(&vs), sizeof(vs));
        for (size_t i = 0; i < vs; ++i) {
            size_t len;
            in.read(reinterpret_cast<char *>(&len), sizeof(len));
            string tok(len, '\0');
            in.read(&tok[0], len);
            token_to_id[tok] = i;
            id_to_token.push_back(tok);
        }
        size_t ms;
        in.read(reinterpret_cast<char *>(&ms), sizeof(ms));
        for (size_t i = 0; i < ms; ++i) {
            size_t fl, sl;
            in.read(reinterpret_cast<char *>(&fl), sizeof(fl));
            string f(fl, '\0');
            in.read(&f[0], fl);
            in.read(reinterpret_cast<char *>(&sl), sizeof(sl));
            string s(sl, '\0');
            in.read(&s[0], sl);
            merges.push_back({f, s});
        }
        in.read(reinterpret_cast<char *>(&use_bpe), sizeof(use_bpe));
    }

  private:
    unordered_map<string, int> token_to_id;
    vector<string> id_to_token;
    vector<pair<string, string>> merges;
    bool use_bpe = false;

    void encode_word(const string &word, vector<int> &ids) {
        if (!use_bpe) {
            ids.push_back(add_token(word));
            return;
        }

        vector<string> splits;
        for (size_t i = 0; i < word.size(); ++i) {
            string ch(1, word[i]);
            if (i == word.size() - 1)
                ch += "</w>";
            splits.push_back(ch);
        }

        for (const auto &[first, second] : merges) {
            vector<string> new_splits;
            for (size_t i = 0; i < splits.size(); ++i) {
                if (i + 1 < splits.size() && splits[i] == first &&
                    splits[i + 1] == second) {
                    new_splits.push_back(first + second);
                    ++i;
                } else {
                    new_splits.push_back(splits[i]);
                }
            }
            splits = new_splits;
            if (splits.size() == 1)
                break;
        }

        for (const auto &s : splits) {
            ids.push_back(get_token(s));
        }
    }
};
