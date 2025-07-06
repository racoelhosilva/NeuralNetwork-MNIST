#include "Matrix.h"
#include <format>
#include <iomanip>
#include <stdexcept>

Matrix Matrix::row(int index) const {
    index = validate_row(index);
    Matrix row {1, m_cols};
    for (int col { 0 }; col < m_cols; ++col) {
        row[0, col] = (*this)[index, col];
    }
    return row;
}

Matrix Matrix::rows(int start, int end) const {
    validate_row_range(start, end);
    Matrix slice {end - start, m_cols};
    for (int row { start }; row < end; ++row) {
        for (int col { 0 }; col < m_cols; ++col) {
            slice[row - start, col] = (*this)[row, col];
        }
    }
    return slice;
}

Matrix Matrix::col(int index) const {
    index = validate_col(index);
    Matrix col {m_rows, 1};
    for (int row { 0 }; row < m_rows; ++row) {
        col[row, 0] = (*this)[row, index];
    }
    return col;
}

Matrix Matrix::cols(int start, int end) const {
    validate_col_range(start, end);
    Matrix slice {m_rows, end - start};
    for (int row { 0 }; row < m_rows; ++row) {
        for (int col { start }; col < end; ++col) {
            slice[row, col - start] = (*this)[row, col];
        }
    }
    return slice;
}

Matrix Matrix::slice(int row_start, int row_end, int col_start, int col_end) const {
    validate_row_range(row_start, row_end);
    validate_col_range(col_start, col_end);
    Matrix slice {row_end - row_start, col_end - col_start};
    for (int row { row_start }; row < row_end; ++row) {
        for (int col { col_start }; col < col_end; ++col) {
            slice[row - row_start, col - col_start] = (*this)[row, col];
        }
    }
    return slice;
}

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

Matrix& Matrix::operator/=(double scalar) {
    for (auto& element : m_data) {
        element /= scalar;
    }
    return *this;
}

Matrix operator/(Matrix matrix, double scalar) {
    return matrix /= scalar;
}

Matrix Matrix::operator-() const {
    Matrix m { *this };
    for (auto& element : m.m_data) {
        element = -element;
    }
    return m;
}

[[nodiscard]] bool operator==(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows()) { return false; }
    if (lhs.cols() != rhs.cols()) { return false; }
    return lhs.m_data == rhs.m_data;
}

[[nodiscard]] bool operator!=(const Matrix& lhs, const Matrix& rhs) {
    return !(lhs == rhs);
}

Matrix Matrix::flatten(bool col) const {    
    Matrix flattened { 
        col ? m_rows * m_cols : 1, 
        col ? 1 : m_rows * m_cols 
    };
    std::copy(m_data.begin(), m_data.end(), flattened.m_data.begin());
    return flattened;
}

Matrix Matrix::transpose() const {
    Matrix transposed { m_cols, m_rows };
    for (int row { 0 }; row < m_rows; ++row) {
        for (int col { 0 }; col < m_cols; ++col) {
            transposed[col, row] = (*this)[row, col];
        }
    }
    return transposed;
}

Matrix Matrix::hadamard(const Matrix& matrix) const {
    check_matching_dimensions(matrix);
    Matrix product { *this };
    for (int row { 0 }; row < m_rows; ++row) {
        for (int col { 0 }; col < m_cols; ++col) {
            product[row, col] *= matrix[row, col];
        }
    }
    return product;
}

Matrix Matrix::matmul(const Matrix& matrix) const {
    check_mult_dimensions(matrix);
    Matrix product { rows(), matrix.cols(), 0.0 };
    for (int i { 0 }; i < m_rows; ++i) {
        for (int k { 0 }; k < m_cols; ++k) {
            const double aik = (*this)[i, k];
            for (int j {0}; j < matrix.cols(); ++j) {
                product[i, j] += aik * matrix[k, j];
            }
        }
    }
    return product;
}

Matrix Matrix::hadamard_div(const Matrix& matrix) const {
    check_matching_dimensions(matrix);
    Matrix product { *this };
    for (int row { 0 }; row < m_rows; ++row) {
        for (int col { 0 }; col < m_cols; ++col) {
            product[row, col] /= matrix[row, col];
        }
    }
    return product;
}

Matrix Matrix::row_avg() const {
    Matrix column {m_rows, 1, 0.0};
    for (int row { 0 }; row < m_rows; ++row) {
        for (int col { 0 }; col < m_cols; ++col) {
            column[row, 0] += (*this)[row, col];
        }
    }
    return column / m_cols;
}

Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
    return lhs.matmul(rhs);
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

void Matrix::validate_row_range(int start, int end) const {
    if (start >= end) {
        throw std::invalid_argument(std::format(
            "invalid range specification [{},{}]",
            start, end
        ));
    }
    if (start < 0 || end > m_rows) {
        throw std::out_of_range(std::format(
            "row range [{},{}] is out of bounds [0, {}]",
            start, end, m_rows - 1
        ));
    }
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

void Matrix::validate_col_range(int start, int end) const {
    if (start >= end) {
        throw std::invalid_argument(std::format(
            "invalid range specification [{},{}]",
            start, end
        ));
    }
    if (start < 0 || end > m_cols) {
        throw std::out_of_range(std::format(
            "col range [{},{}] is out of bounds [0, {}]",
            start, end, m_cols - 1
        ));
    }
}

void Matrix::check_matching_dimensions(const Matrix& other) const {
    if (rows() != other.rows() || cols() != other.cols()) {
        throw std::invalid_argument(std::format(
            "mismatched matrix dimensions ({}x{}) must be ({}x{})",
            other.rows(), other.cols(), rows(), cols()
        ));
    } 
}

void Matrix::check_mult_dimensions(const Matrix& other) const {
    if (cols() != other.rows()) {
        throw std::invalid_argument(std::format(
            "mismatched matrix multiplication inner dimensions ({} and {})",
            cols(), other.rows()
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
