#include "Matrix.h"
#include <format>
#include <stdexcept>

int Matrix::valid_dimension(int dim) {
    if (dim <= 0){
        throw std::invalid_argument(std::format(
            "invalid matrix dimension ({}) must be >= 1",
            dim
        ));
    }
    return dim;
}

int Matrix::valid_row(int row) const {
    if (row < 0 || row >= m_rows) {
        throw std::out_of_range(std::format(
            "row index ({}) is out of bounds [0, {}]",
            row, m_rows - 1
        ));
    }
    return row;
}

int Matrix::valid_col(int col) const {
    if (col < 0 || col >= m_cols) {
        throw std::out_of_range(std::format(
            "col index ({}) is out of bounds [0, {}]",
            col, m_cols - 1
        ));
    }
    return col;
}
