#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

namespace activation {
    double ReLU(double val);
    double ReLU_prime(double val);
    Matrix softmax(const Matrix& logits);
}

#endif
