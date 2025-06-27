#include "Matrix.h"
#include <format>
#include <iomanip>
#include <stdexcept>

void Matrix::fill(double value) noexcept {
    std::fill(m_data.begin(), m_data.end(), value);
}

Matrix& Matrix::operator+=(const Matrix &rhs) {
    check_matching_dimensions(rhs);
    for (size_t idx { 0 }; idx < m_data.size(); ++idx) {
        m_data[idx] += rhs.m_data[idx];
    }
    return *this;
}

Matrix operator+(Matrix m1, const Matrix& m2) {
    return m1 += m2;
}

Matrix& Matrix::operator-=(const Matrix &rhs) {
    check_matching_dimensions(rhs);
    for (size_t idx { 0 }; idx < m_data.size(); ++idx) {
        m_data[idx] -= rhs.m_data[idx];
    }
    return *this;
}

Matrix operator-(Matrix m1, const Matrix& m2) {
    return m1 -= m2;
}

Matrix& Matrix::operator*=(double scalar) noexcept {
    for (auto& element : m_data) {
        element *= scalar;
    }
    return *this;
}

Matrix operator*(Matrix matrix, double scalar) {
    return matrix *= scalar;
}

Matrix operator*(double scalar, Matrix matrix) {
    return matrix *= scalar;
}

int Matrix::validate_dimension(int dim) {
    if (dim <= 0){
        throw std::invalid_argument(std::format(
            "invalid matrix dimension ({}) must be >= 1",
            dim
        ));
    }
    return dim;
}

int Matrix::validate_row(int row) const {
    if (row < 0 || row >= m_rows) {
        throw std::out_of_range(std::format(
            "row index ({}) is out of bounds [0, {}]",
            row, m_rows - 1
        ));
    }
    return row;
}

int Matrix::validate_col(int col) const {
    if (col < 0 || col >= m_cols) {
        throw std::out_of_range(std::format(
            "col index ({}) is out of bounds [0, {}]",
            col, m_cols - 1
        ));
    }
    return col;
}

void Matrix::check_matching_dimensions(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument(std::format(
            "unmatched matrix dimensions ({}x{}) must be ({}x{})",
            other.rows(), other.cols(), rows(), cols()
        ));
    } 
}

std::ostream& operator<<(std::ostream& out, const Matrix& matrix) {
    out << std::fixed << std::setprecision(3);
    for (int row = 0; row < matrix.rows(); row++) {
        out << "[ ";
        for (int col = 0; col < matrix.cols(); col++) {
            out << matrix[row, col] << " ";
        }
        out << "]\n";
    }
    out << "(" << matrix.rows() << " rows, " << matrix.cols() << " cols)\n";
    return out;
}
