#include "Optimizer.h"

std::shared_ptr<optimizer::Base> optimizer::create(
    int rows, int cols, 
    const optimizer::settings& optimizer
) {
    switch (optimizer.type) {
        case Type::SGD:
            return std::make_shared<optimizer::SGD>();
        default:
            throw std::invalid_argument("unknown optimizer type");
    }
}

void optimizer::SGD::update(Matrix& param, const Matrix& grad, double lr) {
    param -= lr * grad;
}


