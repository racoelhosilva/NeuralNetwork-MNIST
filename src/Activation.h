#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

namespace activation {

    enum class Type {
        ReLU,
        Softmax
    };

    Matrix apply(const Matrix& matrix, activation::Type type);
    Matrix apply_prime(const Matrix& matrix, activation::Type type);

    double ReLU(double val);
    double ReLU_prime(double val);
    Matrix softmax(const Matrix& logits);
}

#endif
