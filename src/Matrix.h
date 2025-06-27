#ifndef MATRIX_H
#define MATRIX_H

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

private:
    const int m_rows;
    const int m_cols;
    std::vector<double> m_data;

    static int valid_dimension(int dim);
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

#endif
