#ifndef LOSS_H
#define LOSS_H

#include "Activation.h"
#include "Matrix.h"

namespace loss {

    const double EPSILON = 1e-8;

    enum class Type {
        CrossEntropy,
        MSE
    };

    double compute(const Matrix& label,
        const Matrix& prediction,
        loss::Type type
    );

    Matrix gradient(const Matrix &label, 
        const Matrix &prediction, 
        const Matrix &z, 
        loss::Type type, 
        activation::Type activation
    );
}

#endif
