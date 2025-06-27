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
