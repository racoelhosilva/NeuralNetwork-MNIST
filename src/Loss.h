#ifndef LOSS_H
#define LOSS_H

#include "Activation.h"
#include "Matrix.h"

namespace loss {

    enum class Type {
        CrossEntropy,
        MSE
    };

    Matrix gradient(const Matrix &label, 
        const Matrix &prediction, 
        const Matrix &z, 
        loss::Type type, 
        activation::Type activation
    );
}

#endif
