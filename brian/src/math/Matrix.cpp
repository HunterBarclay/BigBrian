#include "brian/math/Matrix.h"

#include <cassert>
#include <cstring>
#include <sstream>
#include <cstdlib>

namespace bb {

    Matrix::Matrix(uint rows, uint cols): m_rows(rows), m_cols(cols), m_arr(new Real[rows * cols]) {
        for (uint i = 0; i < rows * cols; ++i) {
            this->m_arr[i] = 0;
        }
    }

    Matrix::Matrix(uint rows, uint cols, const Real *const p_data): m_rows(rows), m_cols(cols), m_arr(new Real[rows * cols]) {
        memcpy(this->m_arr, p_data, sizeof(Real) * rows * cols);
    }

    Matrix::~Matrix() {
        delete[] this->m_arr;
    }

    std::unique_ptr<Matrix> Matrix::Mult(const Matrix& p_b) const {
        auto res = std::make_unique<Matrix>(this->m_rows, p_b.m_cols);
        this->Mult(p_b, *res);
        return std::move(res);
    }

    void Matrix::Mult(const Matrix& p_b, Matrix& p_out) const {
        assert(this->m_cols == p_b.m_rows);
        assert(p_out.m_rows == this->m_rows);
        assert(p_out.m_cols == p_b.m_cols);

        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < p_b.m_cols; ++c) {
                Real elem = 0;
                for (uint i = 0; i < this->m_cols; ++i) {
                    elem += this->get(r, i) * p_b.get(i, c);
                }
                p_out.set(r, c, elem);
            }
        }
    }

    std::unique_ptr<Matrix> Matrix::Add(const Matrix& p_b) const {
        auto res = std::make_unique<Matrix>(this->m_rows, p_b.m_cols);
        this->Add(p_b, *res);
        return std::move(res);
    }

    void Matrix::Add(const Matrix& p_b, Matrix& p_out) const {
        assert(this->m_rows == p_b.m_rows);
        assert(this->m_rows == p_out.m_rows);
        assert(this->m_cols == p_b.m_cols);
        assert(this->m_cols == p_out.m_cols);

        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < this->m_cols; ++c) {
                p_out.set(r, c, this->get(r, c) + p_b.get(r, c));
            }
        }
    }

    void Matrix::Mutate(Real (*p_func) (Real)) {
        for (uint r = 0; r < this->m_rows; ++r) {
            for (uint c = 0; c < this->m_cols; ++c) {
                this->set(r, c, p_func(this->get(r, c)));
            }
        }
    }

    void Matrix::setAll(const Real* const p_data) {
        memcpy(this->m_arr, p_data, sizeof(Real) * this->m_rows * this->m_cols);
    }

    const Real* const Matrix::getRef() {
        return this->m_arr;
    }

    Real* Matrix::getCopy() {
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
