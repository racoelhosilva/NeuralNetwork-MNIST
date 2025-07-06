#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Matrix.h"
#include <memory>

namespace optimizer {

    enum class Type {
        SGD,
        Momentum,
        RMSProp,
        Adam
    };

    struct settings {
        optimizer::Type type;
        double epsilon = 1e-8;
        double beta1;
        double beta2;
    };

    class Base {
    public:
        virtual ~Base() = default;
        virtual void update(Matrix& m, const Matrix& grad, double lr) = 0;
    };

    class SGD : public Base {
    public:
        void update(Matrix& param, const Matrix& grad, double lr) override;
    };

    std::shared_ptr<optimizer::Base> create(
        int rows, int cols, 
        const optimizer::settings& optimizer
    );
}

#endif
