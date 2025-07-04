#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <random>
#include <vector>

class Matrix {
public:
    Matrix() = delete;

    explicit Matrix(int rows, int cols, double init = 0.0)
        : m_rows {validate_dimension(rows)}
        , m_cols {validate_dimension(cols)}
        , m_data(
            static_cast<size_t>(rows) * static_cast<size_t>(cols), 
            init
        ) 
    {}

    explicit Matrix(int rows, int cols, std::vector<double>&& data)
        : m_rows {validate_dimension(rows)}
        , m_cols {validate_dimension(cols)}
        , m_data {std::move(data)} 
    {
        if (m_data.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
            throw std::invalid_argument("unmatched row/col and data matrix size");
        }
    }

    ~Matrix() = default;

    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;

    Matrix& operator=(const Matrix& matrix) {
        if (this != &matrix) [[likely]] {
            m_rows = matrix.m_rows;
            m_cols = matrix.m_cols;
            m_data = matrix.m_data;
        }
        return *this;
    }

    Matrix& operator=(Matrix&& matrix) {
        if (this != &matrix) [[likely]] {
            m_rows = matrix.m_rows;
            m_cols = matrix.m_cols;
            m_data = std::move(matrix.m_data);
        }
        return *this;
    }

    [[nodiscard]] constexpr int rows() const noexcept;
    [[nodiscard]] constexpr int cols() const noexcept;
    [[nodiscard]] constexpr std::pair<int, int> shape() const noexcept;

    [[nodiscard]] Matrix row(int index) const;
    [[nodiscard]] Matrix rows(int start, int end) const;
    [[nodiscard]] Matrix col(int index) const;
    [[nodiscard]] Matrix cols(int start, int end) const;
    [[nodiscard]] Matrix slice(int row_start, int row_end, int col_start, int col_end) const;

    [[nodiscard]] double& operator[](int row, int col) noexcept;
    [[nodiscard]] const double& operator[](int row, int col) const noexcept;
    [[nodiscard]] double& at(int row, int col);
    [[nodiscard]] const double& at(int row, int col) const;
    void fill(double value) noexcept;

    Matrix& operator+=(const Matrix& rhs);
    Matrix& operator-=(const Matrix& rhs);
    Matrix& operator*=(double scalar) noexcept;
    Matrix& operator/=(double scalar);
    Matrix operator-() const;

    friend bool operator==(const Matrix& lhs, const Matrix& rhs);
    friend bool operator!=(const Matrix& lhs, const Matrix& rhs);

    [[nodiscard]] Matrix flatten(bool col=true) const;
    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] Matrix hadamard(const Matrix& matrix) const;
    [[nodiscard]] Matrix matmul(const Matrix& matrix) const;

    [[nodiscard]] Matrix row_avg() const;
    
    template <typename Function>
    [[nodiscard]] Matrix apply(Function&& f) const;
private:
    int m_rows;
    int m_cols;
    std::vector<double> m_data;

    [[nodiscard]] static int validate_dimension(int dim);
    [[nodiscard]] int validate_row(int row) const;
    [[nodiscard]] int validate_col(int col) const;
    void validate_row_range(int start, int end) const;
    void validate_col_range(int start, int end) const;
    void check_matching_dimensions(const Matrix& other) const;
    void check_mult_dimensions(const Matrix& other) const;
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

[[nodiscard]] Matrix operator+(Matrix lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator-(Matrix lhs, const Matrix& rhs);
[[nodiscard]] Matrix operator*(Matrix matrix, double scalar);
[[nodiscard]] Matrix operator*(double scalar, Matrix matrix);
[[nodiscard]] Matrix operator/(Matrix matrix, double scalar);
[[nodiscard]] Matrix operator*(const Matrix& lhs, const Matrix& rhs);

std::ostream& operator<<(std::ostream& out, const Matrix& matrix);

#endif
