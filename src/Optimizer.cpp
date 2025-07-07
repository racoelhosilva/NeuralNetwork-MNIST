#include "Optimizer.h"
#include <cmath>

std::shared_ptr<optimizer::Base> optimizer::create(
    int rows, int cols, 
    const optimizer::settings& optimizer
) {
    switch (optimizer.type) {
        case Type::SGD:
            return std::make_shared<optimizer::SGD>();
        case Type::Momentum:
            return std::make_shared<optimizer::Momentum>(rows, cols, optimizer.beta1);
        case Type::RMSProp:
            return std::make_shared<optimizer::RMSProp>(rows, cols, optimizer.beta2, optimizer.epsilon);
        case Type::Adam:
            return std::make_shared<optimizer::Adam>(rows, cols, optimizer.beta1, optimizer.beta2, optimizer.epsilon);
        default:
            throw std::invalid_argument("unknown optimizer type");
    }
}

void optimizer::SGD::update(Matrix& param, const Matrix& grad, double lr) {
    param -= lr * grad;
}

void optimizer::Momentum::update(Matrix& param, const Matrix& grad, double lr) {
    velocity = momentum * velocity - lr * grad;
    param += velocity;
}

void optimizer::RMSProp::update(Matrix& param, const Matrix& grad, double lr) {
    cache = decay * cache + (1 - decay) * (grad.hadamard(grad));

    const Matrix denom = cache.apply([this](double x) {return std::sqrt(x) + this->epsilon;});
    
    param -= lr * (grad.hadamard_div(denom));
}

void optimizer::Adam::update(Matrix& param, const Matrix& grad, double lr) {
    ++t;
    cache_p1 *= beta1;
    cache_p2 *= beta2;

    m = beta1 * m + (1 - beta1) * grad;
    v = beta2 * v + (1 - beta2) * (grad.hadamard(grad));

    const Matrix m_ = m / (1 - cache_p1);
    const Matrix v_ = v / (1 - cache_p2);

    const Matrix denom = v_.apply([this](double x) {return std::sqrt(x) + this->epsilon;}); 
    param -= lr * (m_.hadamard_div(denom));
}


