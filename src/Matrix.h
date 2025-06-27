#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <vector>

class Matrix {
public:
    Matrix() = delete;

    explicit Matrix(int rows, int cols, double init = 0.0)
        : m_rows {valid_dimension(rows)}
        , m_cols {valid_dimension(cols)}
        , m_data(
            static_cast<size_t>(rows) * static_cast<size_t>(cols), 
            init
        ) 
    {}

    [[nodiscard]] constexpr int rows() const noexcept;
    [[nodiscard]] constexpr int cols() const noexcept;
    [[nodiscard]] constexpr std::pair<int, int> shape() const noexcept;

    [[nodiscard]] double& operator[](int row, int col) noexcept;
    [[nodiscard]] const double& operator[](int row, int col) const noexcept;
    [[nodiscard]] double& at(int row, int col);
    [[nodiscard]] const double& at(int row, int col) const;

    friend std::ostream& operator<<(std::ostream& out, const Matrix& matrix);
private:
    const int m_rows;
    const int m_cols;
    std::vector<double> m_data;

    [[nodiscard]] static int valid_dimension(int dim);
    [[nodiscard]] int valid_row(int row) const;
    [[nodiscard]] int valid_col(int col) const;
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
    return m_data[index(valid_row(row), valid_col(col))];
}

inline const double& Matrix::at(int row, int col) const {
    return m_data[index(valid_row(row), valid_col(col))];
}

inline constexpr size_t Matrix::index(int row, int col) const noexcept {
    return static_cast<size_t>(row) 
        * static_cast<size_t>(m_cols) 
        + static_cast<size_t>(col);
}

#endif
