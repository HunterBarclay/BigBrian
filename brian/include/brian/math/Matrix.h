#pragma once
/**
 * @file Matrix.h
 * @author Hunter Barclay
 * @brief NxM Matrix
 */

#include "brian/math/types.h"

#include <memory>
#include <string>
#include <cassert>

namespace bb {
    /**
     * @brief N by M dimensional matrix.
     */
    class Matrix {
    private:
        Real* const m_arr;
        const uint m_rows;
        const uint m_cols;

    public:
        /**
         * @brief Construct a new Matrix object. Default values for elements is 0.
         * 
         * @param rows Number of rows (N).
         * @param cols Number of columns (M).
         */
        Matrix(uint rows, uint cols);
        /**
         * @brief Construct a new Matrix object. Initialize values with array of scalars.
         * 
         * @param rows Number of rows (N).
         * @param cols Number of columns (M).
         * @param data Initial data. Size must be N * M.
         */
        Matrix(uint rows, uint cols, const Real* const data);
        Matrix(const Matrix& _) = delete;
        ~Matrix();

        const std::string str() const;

        void copy_to(Matrix& matrix) const;

        /**
         * @brief Multiply a given matrix (B) by this Matrix (A).
         * A's column count must match B's row count.
         * 
         * Equation -> A * B
         * 
         * @param b Matrix to multiply (B).
         * @return Resulting matrix.
         */
        std::unique_ptr<Matrix> mult(const Matrix& b) const;

        /**
         * @brief Multiply a given matrix (B) by this Matrix (A), and store the
         * result into out (C).
         * 
         * Equation -> C = A * B
         * 
         * A's column count must match B's row count.
         * C's column count must match B's column count.
         * C's row count must match A's row count.
         * 
         * @param b Matrix to multiply (B).
         * @param out Matrix to store results in (C).
         */
        void mult(const Matrix& b, Matrix& out) const;

        void mult(const Real p_coef);

        /**
         * @brief Add a given matrix (B) to this one (A).
         * 
         * Equation -> A + B
         * 
         * @param b Matrix B.
         * @return Resulting matrix.
         */
        std::unique_ptr<Matrix> add(const Matrix& b) const;

        /**
         * @brief Add a given matrix (B) to this one (A), and store the
         * result into out (C).
         * 
         * Equation -> C = A + B
         * 
         * @param b Matrix B.
         * @param out Matrix C.
         */
        void add(const Matrix& b, Matrix& out) const;

        /**
         * @brief Mutate each element in the matrix (A) via given function (F).
         * 
         * Equation A_ij = F(A_ij)
         * 
         * @param func Mutation (F).
         */
        void mutate(Real (*func) (Real));

        void clear();

        /**
         * @brief Sets all elements in the matrix. Data must be same size as the matrix.
         * 
         * @param data Source data.
         */
        void set_all(const Real* const data);

        /**
         * @brief Gets an active reference to all elements in the matrix.
         * 
         * @return Pointer to matrix data.
         */
        const Real* const get_ref();

        /**
         * @brief Gets a copy of the data in the matrix.
         * 
         * @return Pointer with copy of the matrix data.
         */
        Real* get_copy();
        
        /**
         * @brief Set element in the matrix.
         * 
         * @param row Row of element.
         * @param col Column of element.
         * @param value Value to assign.
         */
        inline void set(uint row, uint col, Real value) {
            assert(row >= 0 && row < this->m_rows);
            assert(col >= 0 && col < this->m_cols);
            this->m_arr[row * this->m_cols + col] = value;
        }

        /**
         * @brief Get element in matrix.
         * 
         * @param row Row of element.
         * @param col Column of element.
         * @return Selected element.
         */
        inline Real get(uint row, uint col) const {
            assert(row >= 0 && row < this->m_rows);
            assert(col >= 0 && col < this->m_cols);
            return this->m_arr[row * this->m_cols + col];
        }

        /**
         * @brief Get the number of rows.
         * 
         * @return Number of rows.
         */
        const inline uint get_num_rows() const {
            return this->m_rows;
        }

        /**
         * @brief Get the number of columns.
         * 
         * @return Number of columns.
         */
        const inline uint get_num_cols() const {
            return this->m_cols;
        }
    };
}
