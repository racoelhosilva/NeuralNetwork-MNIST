#include "Loss.h"
#include <stdexcept>

Matrix loss::gradient(const Matrix &label, 
        const Matrix &prediction, 
        const Matrix &z, 
        loss::Type type, 
        activation::Type activation
    ) {
    switch (type) {
        case Type::CrossEntropy:
            if (activation == activation::Type::Softmax
                || activation == activation::Type::Sigmoid) {
                return prediction - label;
            }
            else {
                throw std::invalid_argument("unsupported activation function for cross entropy loss");
            }
        case Type::MSE:
            return (prediction - label)
                .hadamard(activation::apply_prime(z, activation));
        default:
            throw std::invalid_argument("unknown loss function type");
    }
}
