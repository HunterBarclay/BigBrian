#include "math/Matrix.h"

bb::Matrix::Matrix(U32 rows, U32 cols): m_rows(rows), m_cols(cols), m_arr(new Real[rows * cols]) {
    for (U32 i = 0; i < rows * cols; ++i) {
        this->m_arr[i] = 0;
    }
}

bb::Matrix::Matrix(U32 rows, U32 cols, const Real *const data): m_rows(rows), m_cols(cols), m_arr(new Real[rows * cols]) {
    for (U32 i = 0; i < rows * cols; ++i) {
        this->m_arr[i] = data[i];
    }
}

bb::Matrix::~Matrix() {
    delete[] this->m_arr;
}

std::unique_ptr<bb::Matrix> bb::Matrix::Mult(const Matrix& matrix) const {
    assert(this->m_cols == matrix.m_rows);
    
    auto res = std::make_unique<bb::Matrix>(this->m_rows, matrix.m_cols);

    for (U32 r = 0; r < this->m_rows; ++r) {
        for (U32 c = 0; c < matrix.m_cols; ++c) {
            Real elem = 0;
            for (U32 i = 0; i < this->m_cols; ++i) {
                elem += this->get(r, i) * matrix.get(i, c);
            }
            res->set(r, c, elem);
        }
    }

    return res;
}
