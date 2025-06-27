#include "Matrix.h"
#include <stdexcept>

int Matrix::valid_dimension(int dim) {
    if (dim <= 0){
        throw std::invalid_argument("cannot create matrix with dimensions smaller than 1");
    }
    return dim;
}

