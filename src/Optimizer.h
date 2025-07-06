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

    class Momentum : public Base {
    public:
        Momentum(int rows, int cols, double momentum)
            : velocity(rows, cols), momentum {momentum} {}
        void update(Matrix& param, const Matrix& grad, double lr) override;
    private:
        Matrix velocity;
        double momentum;
    };

    class RMSProp : public Base {
    public:
        RMSProp(int rows, int cols, double decay, double epsilon)
            : cache(rows, cols), decay {decay}, epsilon {epsilon} {}
        void update(Matrix& param, const Matrix& grad, double lr) override;
    private:
        Matrix cache;
        double decay;
        double epsilon;
    };

    class Adam : public Base {
    public:
        Adam(int rows, int cols, double beta1, double beta2, double epsilon)
            : m(rows, cols), v(rows, cols)
            , beta1 {beta1}, beta2 {beta2}, epsilon {epsilon}
        {}
        void update(Matrix& param, const Matrix& grad, double lr) override;
    private:
        Matrix m, v;
        double beta1, beta2, epsilon;
        double cache_p1 = 1.0;
        double cache_p2 = 1.0;
        int t = 0;
    };

    std::shared_ptr<optimizer::Base> create(
        int rows, int cols, 
        const optimizer::settings& optimizer
    );
}

#endif
