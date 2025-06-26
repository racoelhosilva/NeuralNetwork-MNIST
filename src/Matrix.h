#ifndef MATRIX_H
#define MATRIX_H

#include <stdexcept>
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
