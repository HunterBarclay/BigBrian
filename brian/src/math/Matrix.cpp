#include "brian/math/matrix.h"

#include <cassert>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <iostream>

namespace bb {

    Matrix::Matrix(uint rows, uint cols): m_rows(rows), m_cols(cols), m_arr(new Real[rows * cols]) {
        for (uint i = 0; i < rows * cols; ++i) {
            this->m_arr[i] = 0;
        }
    }

    Matrix::Matrix(uint rows, uint cols, const Real *const data): m_rows(rows), m_cols(cols), m_arr(new Real[rows * cols]) {
        memcpy(this->m_arr, data, sizeof(Real) * rows * cols);
    }

    Matrix::~Matrix() {
        delete[] this->m_arr;
    }

    void Matrix::copy_to(Matrix& matrix) const {
        assert(this->m_rows == matrix.m_rows);
        assert(this->m_cols == matrix.m_cols);
        memcpy(matrix.m_arr, this->m_arr, sizeof(Real) * this->m_rows * this->m_cols);
    }

    std::unique_ptr<Matrix> Matrix::mult(const Matrix& b) const {
        auto res = std::make_unique<Matrix>(this->m_rows, b.m_cols);
        this->mult(b, *res);
        return std::move(res);
    }

    void Matrix::mult(const Matrix& b, Matrix& out) const {
        assert(this->m_cols == b.m_rows);
        assert(out.m_rows == this->m_rows);
        assert(out.m_cols == b.m_cols);

        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < b.m_cols; ++c) {
                Real elem = 0;
                for (uint i = 0; i < this->m_cols; ++i) {
                    elem += this->get(r, i) * b.get(i, c);
                }
                out.set(r, c, elem);
            }
        }
    }

    void Matrix::mult(const Real coef) {
        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < this->m_cols; ++c) {
                this->set(r, c, this->get(r, c) * coef);
            }
        }
    }

    std::unique_ptr<Matrix> Matrix::add(const Matrix& b) const {
        auto res = std::make_unique<Matrix>(this->m_rows, b.m_cols);
        this->add(b, *res);
        return std::move(res);
    }

    void Matrix::add(const Matrix& b, Matrix& out) const {
        assert(this->m_rows == b.m_rows);
        assert(this->m_rows == out.m_rows);
        assert(this->m_cols == b.m_cols);
        assert(this->m_cols == out.m_cols);

        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < this->m_cols; ++c) {
                out.set(r, c, this->get(r, c) + b.get(r, c));
            }
        }
    }

    void Matrix::mutate(Real (*func) (Real)) {
        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < this->m_cols; ++c) {
                this->set(r, c, func(this->get(r, c)));
            }
        }
    }

    void Matrix::clear() {
        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < this->m_cols; ++c) {
                this->set(r, c, 0);
            }
        }
    }

    void Matrix::set_all(const Real* const data) {
        memcpy(this->m_arr, data, sizeof(Real) * this->m_rows * this->m_cols);
    }

    const Real* const Matrix::get_ref() {
        return this->m_arr;
    }

    Real* Matrix::get_copy() {
        auto out = (Real*) malloc(sizeof(Real) * this->m_rows * this->m_cols);
        memcpy(out, this->m_arr, sizeof(Real) * this->m_rows * this->m_cols);
        return out;
    }

    const std::string Matrix::str() const {
        static char buff[10];
        std::stringstream ss;

        sprintf(buff, "Matrix %u x %u [\n", this->m_rows, this->m_cols);
        ss << buff;
        for (uint r = 0; r < this->m_rows; ++r) {
            ss << "\t";
            for (uint c = 0; c < this->m_cols; ++c) {
                sprintf(buff, "%5.3g, ", this->get(r, c));
                ss << buff;
            }
            ss << "\n";
        }
        ss << "]\n";

        return ss.str();
    }

}
