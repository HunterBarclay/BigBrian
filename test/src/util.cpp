#include "util.h"

void validate_matrix(const bb::Matrix& mat, const uint rows, const uint cols) {
    assert(mat.get_num_rows() == rows);
    assert(mat.get_num_cols() == cols);
}

void validate_matrix(const bb::Matrix& mat, const uint rows, const uint cols, const bb::Real* const data) {
    validate_matrix(mat, rows, cols);
    for (uint r = 0; r < rows; ++r) {
        for (uint c = 0; c < cols; ++c) {
            assert(bb::repsilon(data[r * cols + c], mat.get(r, c)));
        }
    }
}
