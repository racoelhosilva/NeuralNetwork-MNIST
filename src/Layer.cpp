#include "Layer.h"
#include <iostream>

Matrix Layer::predict(const Matrix& a_prev) const {
    return activation::apply(w * a_prev + b, activation);
}

Matrix Layer::forward(const Matrix& a_prev) {
    cached_input = std::move(a_prev);
    z = std::move(w * a_prev + b);
    return activation::apply(z, activation);
}

Matrix Layer::backward(const Matrix& gradient) {
    Matrix delta = gradient
        .hadamard(activation::apply_prime(z, activation));
    
    dw = delta * cached_input.transpose(); // single input, no regularization
    db = delta; // single input

    return w.transpose() * delta;
}

Matrix Layer::loss(const Matrix& label, const Matrix& prediction, loss::Type loss) {
    Matrix delta = loss::gradient(label, 
        prediction, 
        z,
        loss,
        activation
    );
    
    dw = delta * cached_input.transpose(); // single input, no regularization
    db = delta; // single input
    
    return w.transpose() * delta;
}

void Layer::update(double learning_rate) {
    w -= learning_rate * dw;
    b -= learning_rate * db;
}
