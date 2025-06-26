#ifndef MATRIX_H
#define MATRIX_H

#include <stdexcept>
#include <utility>
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

    [[nodiscard]] constexpr int rows() const noexcept { return m_rows; }
    [[nodiscard]] constexpr int cols() const noexcept { return m_cols; }
    [[nodiscard]] constexpr std::pair<int, int> shape() const noexcept {
        return std::pair<int, int> {m_rows, m_cols};
    }

private:
    const int m_rows;
    const int m_cols;
    std::vector<double> m_data;

    static int valid_dimension(int dim) {
        if (dim <= 0){
            throw std::invalid_argument("cannot create matrix with dimensions smaller than 1");
        }

        return dim;
    } 
};

#endif
