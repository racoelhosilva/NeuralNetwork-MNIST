#pragma once

#include <ostream>
#include <vector>

/**
 * @class Matrix
 * @brief Representation of a 2D matrix of doubles.
 */
class Matrix {
public:
    /**
     * @brief Deleted default constructor.
     */
    Matrix() = delete;

    /**
     * @brief Constructs a matrix with the given dimensions and initial value.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param init Initial value for all element (default = 0.0).
     * @throws std::invalid_argument if rows or cols are less than 1.
     */
    explicit Matrix(int rows, int cols, double init = 0.0)
        : m_rows {validate_dimension(rows)}
        , m_cols {validate_dimension(cols)}
        , m_data(
            static_cast<size_t>(rows) * static_cast<size_t>(cols), 
            init
        ) 
    {}

    /**
     * @brief Constructs a matrix with the given dimensions and data.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param data Vector containing the matrix data.
     * @throws std::invalid_argument if rows or cols are less than 1 
     * or if the size of data does not match the specified dimensions.
     */
    explicit Matrix(int rows, int cols, std::vector<double>&& data)
        : m_rows {validate_dimension(rows)}
        , m_cols {validate_dimension(cols)}
        , m_data {std::move(data)} 
    {
        if (m_data.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
            throw std::invalid_argument("unmatched row/col and data matrix size");
        }
    }

    /**
     * @brief Default destructor.
     */
    ~Matrix() = default;

    /**
     * @brief Copy constructor.
     */
    Matrix(const Matrix&) = default;

    /**
     * @brief Move constructor.
     */
    Matrix(Matrix&&) = default;

    /**
     * @brief Copy assignment operator.
     */
    Matrix& operator=(const Matrix& matrix) noexcept {
        if (this != &matrix) [[likely]] {
            m_rows = matrix.m_rows;
            m_cols = matrix.m_cols;
            m_data = matrix.m_data;
        }
        return *this;
    }

    /**
     * @brief Move assignment operator.
     */
    Matrix& operator=(Matrix&& matrix) noexcept {
        if (this != &matrix) [[likely]] {
            m_rows = matrix.m_rows;
            m_cols = matrix.m_cols;
            m_data = std::move(matrix.m_data);
        }
        return *this;
    }

    /**
     * @brief Returns the number of rows in the matrix.
     * @return Number of rows.
     */
    [[nodiscard]] constexpr int rows() const noexcept;
    
    /**
     * @brief Returns the number of columns in the matrix.
     * @return Number of columns.
     */
    [[nodiscard]] constexpr int cols() const noexcept;
    
    /**
     * @brief Returns the shape of the matrix as a pair (rows, cols).
     * @return Pair containing the number of rows and columns.
     */
    [[nodiscard]] constexpr std::pair<int, int> shape() const noexcept;

    /**
     * @brief Returns a row of the matrix as a new Matrix object.
     * @param index Index of the row to return.
     * @return New Matrix object containing the specified row.
     * @throws std::out_of_range if the index is out of bounds.
     */
    [[nodiscard]] Matrix row(int index) const;

    /**
     * @brief Returns a slice of the matrix containing rows from start to end (exclusive).
     * @param start Starting index of the rows (inclusive).
     * @param end Ending index of the rows (exclusive).
     * @return New Matrix object containing the specified rows.
     * @throws std::out_of_range if the range is invalid or out of bounds.
     * @throws std::invalid_argument if start is greater than or equal to end.
     */
    [[nodiscard]] Matrix rows(int start, int end) const;

    /**
     * @brief Returns a column of the matrix as a new Matrix object.
     * @param index Index of the column to return.
     * @return New Matrix object containing the specified column.
     * @throws std::out_of_range if the index is out of bounds.
     */
    [[nodiscard]] Matrix col(int index) const;

    /**
     * @brief Returns a slice of the matrix containing columns from start to end (exclusive).
     * @param start Starting index of the columns (inclusive).
     * @param end Ending index of the columns (exclusive).
     * @return New Matrix object containing the specified columns.
     * @throws std::out_of_range if the range is invalid or out of bounds.
     * @throws std::invalid_argument if start is greater than or equal to end.
     */
    [[nodiscard]] Matrix cols(int start, int end) const;

    /**
     * @brief Returns a slice of the matrix defined by the specified row and column ranges.
     * @param row_start Starting index of the rows (inclusive).
     * @param row_end Ending index of the rows (exclusive).
     * @param col_start Starting index of the columns (inclusive).
     * @param col_end Ending index of the columns (exclusive).
     * @return New Matrix object containing the specified slice.
     * @throws std::out_of_range if the specified ranges are invalid or out of bounds
     * @throws std::invalid_argument if row_start is greater than or equal to row_end
     * or col_start is greater than or equal to col_end.
     */
    [[nodiscard]] Matrix slice(int row_start, int row_end, int col_start, int col_end) const;

    /**
     * @brief Accesses an element at the specified row and column indices.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at the specified position.
     * @note This method does not perform bounds checking.
     */
    [[nodiscard]] double& operator[](int row, int col) noexcept;

    /**
     * @brief Accesses an element at the specified row and column indices (const version).
     * @param row Row index.
     * @param col Column index.
     * @return Const reference to the element at the specified position.
     * @note This method does not perform bounds checking.
     */
    [[nodiscard]] const double& operator[](int row, int col) const noexcept;

    /**
     * @brief Accesses an element at the specified row and column indices with bounds checking.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the element at the specified position.
     * @throws std::out_of_range if the row or column index is out of bounds
     */
    [[nodiscard]] double& at(int row, int col);

    /**
     * @brief Accesses an element at the specified row and column indices with bounds checking (const version).
     * @param row Row index.
     * @param col Column index.
     * @return Const reference to the element at the specified position.
     * @throws std::out_of_range if the row or column index is out of bounds
     */
    [[nodiscard]] const double& at(int row, int col) const;

    /**
     * @brief Fills the matrix with the specified value.
     * @param value Value to fill the matrix with.
     */
    void fill(double value) noexcept;

    /**
     * @brief Adds another matrix to this matrix.
     * @param rhs The matrix to add.
     * @return Reference to this matrix after addition.
     * @throws std::invalid_argument if the dimensions of the matrices do not match.
     */
    Matrix& operator+=(const Matrix& rhs);

    /**
     * @brief Subtracts another matrix from this matrix.
     * @param rhs The matrix to subtract.
     * @return Reference to this matrix after subtraction.
     * @throws std::invalid_argument if the dimensions of the matrices do not match.
     */
    Matrix& operator-=(const Matrix& rhs);

    /**
     * @brief Multiplies this matrix by a scalar.
     * @param scalar The scalar to multiply by.
     * @return Reference to this matrix after multiplication.
     * @note This operation is performed element-wise.
     */
    Matrix& operator*=(double scalar) noexcept;

    /**
     * @brief Divides this matrix by a scalar.
     * @param scalar The scalar to divide by.
     * @return Reference to this matrix after division.
     * @throws std::invalid_argument if the scalar is zero.
     * @note This operation is performed element-wise.
     */
    Matrix& operator/=(double scalar);

    /**
     * @brief Negates this matrix.
     * @return A new matrix that is the negation of this matrix.
     * @note This operation is performed element-wise.
     */
    Matrix operator-() const;

    /**
     * @brief Compares two matrices for equality.
     * @param lhs The left-hand side matrix.
     * @param rhs The right-hand side matrix.
     * @return True if the matrices are equal, false otherwise.
     * @note Two matrices are considered equal if they have the same dimensions and 
     * all corresponding elements are equal.
     */
    friend bool operator==(const Matrix& lhs, const Matrix& rhs) noexcept;

    /**
     * @brief Compares two matrices for inequality.
     * @param lhs The left-hand side matrix.
     * @param rhs The right-hand side matrix.
     * @return True if the matrices are not equal, false otherwise.
     * @note Two matrices are considered not equal if they differ in dimensions or 
     * any corresponding elements are not equal.
     */
    friend bool operator!=(const Matrix& lhs, const Matrix& rhs) noexcept;

    /**
     * @brief Flattens the matrix into a single row or column.
     * @param col If true, flattens to a single column; if false, to a single row.
     * @return A new Matrix object containing the flattened data.
     */
    [[nodiscard]] Matrix flatten(bool col=true) const;

    /**
     * @brief Transposes the matrix.
     * @return A new Matrix object that is the transpose of this matrix.
     */
    [[nodiscard]] Matrix transpose() const;

    /**
     * @brief Returns a new matrix that is the Hadamard product of this matrix and another matrix.
     * @param matrix The matrix to multiply with.
     * @return A new Matrix object containing the Hadamard product.
     * @throws std::invalid_argument if the dimensions of the matrices do not match.
     */
    [[nodiscard]] Matrix hadamard(const Matrix& matrix) const;

    /**
     * @brief Multiplies this matrix by another matrix.
     * @param matrix The matrix to multiply with.
     * @return A new Matrix object containing the product.
     * @throws std::invalid_argument if the inner dimensions of the matrices do not match.
     */
    [[nodiscard]] Matrix matmul(const Matrix& matrix) const;

    /**
     * @brief Returns a new matrix that is the Hadamard division of this matrix by another matrix.
     * @param matrix The matrix to divide by.
     * @return A new Matrix object containing the Hadamard division.
     * @throws std::invalid_argument if the dimensions of the matrices do not match 
     * or if any element in the other matrix is zero.
     * @note This operation divides each element of this matrix by the corresponding element of the other matrix.
     */
    [[nodiscard]] Matrix hadamard_div(const Matrix& matrix) const;

    /**
     * @brief Computes the average of each row in the matrix.
     * @return A new Matrix object containing the average of each row.
     */
    [[nodiscard]] Matrix row_avg() const;

    /**
     * @brief Computes the average of each column in the matrix.
     * @return A new Matrix object containing the average of each column.
     */
    [[nodiscard]] Matrix col_avg() const;
    
    /**
     * @brief Applies a function to each element of the matrix and returns a new matrix with the results.
     * @param f The function to apply to each element.
     * @return A new Matrix object containing the results of applying the function.
     * @note The function should take a double and return a double.
     */
    template <typename Function>
    [[nodiscard]] Matrix apply(Function&& f) const;
private:
    /**
     * @brief Number of rows in the matrix.
     */
    int m_rows;

    /**
     * @brief Number of columns in the matrix.
     */
    int m_cols;

    /**
     * @brief Vector containing the matrix data in row-major order.
     */
    std::vector<double> m_data;

    /**
     * @brief Validates the dimension of a matrix.
     * @param dim The dimension to validate.
     * @return The validated dimension.
     * @throws std::invalid_argument if the dimension is less than 1.
     */
    [[nodiscard]] static int validate_dimension(int dim);

    /**
     * @brief Validates a row index.
     * @param row The row index to validate.
     * @return The validated row index.
     * @throws std::out_of_range if the row index is out of bounds.
     */
    [[nodiscard]] int validate_row(int row) const;

    /**
     * @brief Validates a column index.
     * @param col The column index to validate.
     * @return The validated column index.
     * @throws std::out_of_range if the column index is out of bounds.
     */
    [[nodiscard]] int validate_col(int col) const;

    /**
     * @brief Validates a range of rows.
     * @param start Starting index of the range (inclusive).
     * @param end Ending index of the range (exclusive).
     * @throws std::out_of_range if the range is invalid or out of bounds.
     * @throws std::invalid_argument if start is greater than or equal to end.
     */
    void validate_row_range(int start, int end) const;

    /**
     * @brief Validates a range of columns.
     * @param start Starting index of the range (inclusive).
     * @param end Ending index of the range (exclusive).
     * @throws std::out_of_range if the range is invalid or out of bounds.
     * @throws std::invalid_argument if start is greater than or equal to end.
     */
    void validate_col_range(int start, int end) const;

    /**
     * @brief Checks if the dimensions of two matrices match.
     * @param other The other matrix to compare against.
     * @throws std::invalid_argument if the dimensions do not match.
     */
    void check_matching_dimensions(const Matrix& other) const;

    /**
     * @brief Checks if the dimensions of two matrices match for element-wise operations.
     * @param other The other matrix to compare against.
     * @throws std::invalid_argument if the dimensions do not match.
     */
    void check_mult_dimensions(const Matrix& other) const;

    /**
     * @brief Computes the index in the data vector for the given row and column.
     * @param row The row index.
     * @param col The column index.
     * @return The computed index.
     */
    [[nodiscard]] constexpr std::size_t index(int row, int col) const noexcept;
};

inline constexpr int Matrix::rows() const noexcept {
    return m_rows;
}

inline constexpr int Matrix::cols() const noexcept {
    return m_cols;
}

inline constexpr std::pair<int, int> Matrix::shape() const noexcept {
    return std::pair<int, int> {m_rows, m_cols};
}

inline double& Matrix::operator[](int row, int col) noexcept {
    return m_data[index(row, col)];
}

inline const double& Matrix::operator[](int row, int col) const noexcept { 
    return m_data[index(row, col)];
}

inline double& Matrix::at(int row, int col) {
    return m_data[index(validate_row(row), validate_col(col))];
}

inline const double& Matrix::at(int row, int col) const {
    return m_data[index(validate_row(row), validate_col(col))];
}

inline constexpr size_t Matrix::index(int row, int col) const noexcept {
    return static_cast<size_t>(row) 
        * static_cast<size_t>(m_cols) 
        + static_cast<size_t>(col);
}

template <typename Function>
inline Matrix Matrix::apply(Function&& f) const {
    Matrix result { *this };
    for (auto& x : result.m_data) {
        x = std::forward<Function>(f)(x);
    }
    return result;
}

/**
 * @brief Adds two matrices element-wise.
 * @param lhs The left-hand side matrix.
 * @param rhs The right-hand side matrix.
 * @return A new Matrix object containing the result of the addition.
 * @throws std::invalid_argument if the dimensions of the matrices do not match.
 */
[[nodiscard]] Matrix operator+(Matrix lhs, const Matrix& rhs);

/**
 * @brief Subtracts two matrices element-wise.
 * @param lhs The left-hand side matrix.
 * @param rhs The right-hand side matrix.
 * @return A new Matrix object containing the result of the subtraction.
 * @throws std::invalid_argument if the dimensions of the matrices do not match.
 */
[[nodiscard]] Matrix operator-(Matrix lhs, const Matrix& rhs);

/**
 * @brief Multiplies a matrix by a scalar.
 * @param matrix The matrix to multiply.
 * @param scalar The scalar to multiply by.
 * @return A new Matrix object containing the result of the multiplication.
 */
[[nodiscard]] Matrix operator*(Matrix matrix, double scalar);

/**
 * @brief Multiplies a scalar by a matrix.
 * @param scalar The scalar to multiply by.
 * @param matrix The matrix to multiply.
 * @return A new Matrix object containing the result of the multiplication.
 */
[[nodiscard]] Matrix operator*(double scalar, Matrix matrix);

/**
 * @brief Divides a matrix by a scalar.
 * @param matrix The matrix to divide.
 * @param scalar The scalar to divide by.
 * @return A new Matrix object containing the result of the division.
 * @throws std::invalid_argument if the scalar is zero.
 */
[[nodiscard]] Matrix operator/(Matrix matrix, double scalar);

/**
 * @brief Multiplies two matrices using matrix multiplication.
 * @param lhs The left-hand side matrix.
 * @param rhs The right-hand side matrix.
 * @return A new Matrix object containing the result of the multiplication.
 * @throws std::invalid_argument if the inner dimensions of the matrices do not match.
 */
[[nodiscard]] Matrix operator*(const Matrix& lhs, const Matrix& rhs);

/**
 * @brief Outputs the matrix to an output stream.
 * @param out The output stream to write to.
 * @param matrix The matrix to output.
 * @return The output stream after writing the matrix.
 */
std::ostream& operator<<(std::ostream& out, const Matrix& matrix);
