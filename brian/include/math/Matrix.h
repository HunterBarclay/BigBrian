#pragma once

#include "math/Types.h"

#include <memory>

namespace bb {
    class Matrix {
    private:
        Real* const m_arr;
        const U32 m_rows;
        const U32 m_cols;

    public:
        Matrix(U32 rows, U32 cols);
        Matrix(U32 rows, U32 cols, const Real* const data);
        Matrix(const Matrix& _) = delete;
        ~Matrix();

        std::unique_ptr<Matrix> Mult(const Matrix& matrix) const;
        inline void set(U32 row, U32 col, Real value) {
            this->m_arr[row * this->m_cols + col] = value;
        }
        inline Real get(U32 row, U32 col) const {
            return this->m_arr[row * this->m_cols + col];
        }

        const inline U32 getRows() const {
            return this->m_rows;
        }
        const inline U32 getCols() const {
            return this->m_cols;
        }
    };
}
