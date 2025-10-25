#include "calculations.h"

int vector_dot_product(const vector<int> &a, const vector<int> &b) {
    bool valid_op = a.size() == b.size();

    if (!valid_op) {
        throw invalid_argument("Invalid operation: vector sizes do not match");
    }
    int result = 0;
    for (size_t i = 0; i < a.size(); i++) {
        result += (a[i] * b[i]);
    }
    return result;
}

matrix matrix_dot_product(const matrix &a, const matrix &b) {
    size_t rows_a = a.size();
    size_t cols_a = a[0].size();
    size_t rows_b = b.size();
    size_t cols_b = b[0].size();

    bool valid_op = cols_a == rows_b;

    if (!valid_op) {
        throw invalid_argument("Invalid operation: matrix sizes do not match");
    }
    matrix result(rows_a, vector<int>(cols_b, 0));
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            for (size_t k = 0; k < cols_a; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}
