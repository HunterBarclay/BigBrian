#include "util.h"

void ValidateMatrix(const bb::Matrix& mat, const uint rows, const uint cols) {
    assert(mat.getNumRows() == rows);
    assert(mat.getNumCols() == cols);
}

void ValidateMatrix(const bb::Matrix& mat, const uint rows, const uint cols, const bb::Real* const data) {
    ValidateMatrix(mat, rows, cols);
    for (uint r = 0; r < rows; ++r) {
        for (uint c = 0; c < cols; ++c) {
            assert(bb::repsilon(data[r * cols + c], mat.get(r, c)));
        }
    }
}
