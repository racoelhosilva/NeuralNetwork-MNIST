#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#include "Matrix.h"

namespace regularization {
    enum class Type {
        None,
        L1,
        L2,
        Elastic
    };

    Matrix term(
        const Matrix& w, 
        regularization::Type type, 
        double lambda1,
        double lambda2
    );
}

#endif
