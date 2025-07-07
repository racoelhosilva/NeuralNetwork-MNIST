#pragma once

#include "Matrix.h"

namespace regularization {
    enum class Type {
        None,
        L1,
        L2,
        Elastic
    };

    struct settings {
        regularization::Type type = Type::None;
        double lambda1 = 0.0;
        double lambda2 = 0.0;
    };

    Matrix term(
        const Matrix& w, 
        regularization::settings regularization
    );
}
