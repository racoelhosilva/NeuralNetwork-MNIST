#include "Loss.h"
#include <cmath>
#include <format>
#include <stdexcept>

double loss::compute(
    const Matrix& label,
    const Matrix& prediction,
    loss::Type type
) {
    const int ROWS = label.rows();
    const int COLS = label.cols();
    if (prediction.rows() != ROWS || prediction.cols() != COLS) {
        throw std::invalid_argument(std::format(
            "mismatch between prediction ({} x {}) and label ({} x {}) sizes",
            prediction.rows(), prediction.cols(),
            ROWS, COLS
        ));
    }

    double total = 0.0;
    switch(type) {
        case Type::CrossEntropy: {
            for (int row { 0 }; row < ROWS; ++row) {
                for (int col { 0 }; col < COLS; ++col) {
                    if (label[row, col] > 0.0) {
                        total -= std::log(prediction[row, col] + loss::EPSILON);
                    }
                }
            }
            return total / COLS;
        }
        case Type::MSE: {
            for (int row { 0 }; row < ROWS; ++row) {
                for (int col { 0 }; col < COLS; ++col) {
                    const double diff = prediction[row, col] - label[row, col];
                    total += diff * diff;
                }
            }
            return total / (ROWS * COLS);
        }
        default:
            throw std::invalid_argument("unknown loss function type");    
    }
}

Matrix loss::gradient(
    const Matrix &label, 
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
