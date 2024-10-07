#pragma once
/**
 * @file Matrix.h
 * @author Hunter Barclay
 * @brief NxM Matrix
 */

#include "brian/math/Types.h"

#include <memory>
#include <string>

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
         * @param p_data Initial data. Size must be N * M.
         */
        Matrix(uint rows, uint cols, const Real* const p_data);
        Matrix(const Matrix& _) = delete;
        ~Matrix();

        const std::string str() const;

        /**
         * @brief Multiply a given matrix (B) by this Matrix (A).
         * A's column count must match B's row count.
         * 
         * Equation -> A * B
         * 
         * @param p_b Matrix to multiply (B).
         * @return Resulting matrix.
         */
        std::unique_ptr<Matrix> Mult(const Matrix& p_b) const;

        /**
         * @brief Multiply a given matrix (B) by this Matrix (A), and store the
         * result into p_out (C).
         * 
         * Equation -> C = A * B
         * 
         * A's column count must match B's row count.
         * C's column count must match B's column count.
         * C's row count must match A's row count.
         * 
         * @param p_b Matrix to multiply (B).
         * @param p_out Matrix to store results in (C).
         */
        void Mult(const Matrix& p_b, Matrix& p_out) const;

        /**
         * @brief Add a given matrix (B) to this one (A).
         * 
         * Equation -> A + B
         * 
         * @param p_b Matrix B.
         * @return Resulting matrix.
         */
        std::unique_ptr<Matrix> Add(const Matrix& p_b) const;

        /**
         * @brief Add a given matrix (B) to this one (A), and store the
         * result into p_out (C).
         * 
         * Equation -> C = A + B
         * 
         * @param p_b Matrix B.
         * @param p_out Matrix C.
         */
        void Add(const Matrix& p_b, Matrix& p_out) const;

        /**
         * @brief Mutate each element in the matrix (A) via given function (F).
         * 
         * Equation A_ij = F(A_ij)
         * 
         * @param p_func Mutation (F).
         */
        void Mutate(Real (*p_func) (Real));

        /**
         * @brief Sets all elements in the matrix. Data must be same size as the matrix.
         * 
         * @param p_data Source data.
         */
        void setAll(const Real* const p_data);

        /**
         * @brief Gets an active reference to all elements in the matrix.
         * 
         * @return Pointer to matrix data.
         */
        const Real* const getRef();

        /**
         * @brief Gets a copy of the data in the matrix.
         * 
         * @return Pointer with copy of the matrix data.
         */
        Real* getCopy();
        
        /**
         * @brief Set element in the matrix.
         * 
         * @param row Row of element.
         * @param col Column of element.
         * @param value Value to assign.
         */
        inline void set(uint row, uint col, Real value) {
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
            return this->m_arr[row * this->m_cols + col];
        }

        /**
         * @brief Get the number of rows.
         * 
         * @return Number of rows.
         */
        const inline uint getNumRows() const {
            return this->m_rows;
        }

        /**
         * @brief Get the number of columns.
         * 
         * @return Number of columns.
         */
        const inline uint getNumCols() const {
            return this->m_cols;
        }
    };
}
