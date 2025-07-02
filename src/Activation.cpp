#include "Activation.h"
#include <algorithm>
#include <cmath>

Matrix activation::apply(const Matrix& matrix, activation::Type type) {
    switch (type) {
        case Type::ReLU:
            return matrix.apply(ReLU);
        case Type::Sigmoid:
            return matrix.apply(sigmoid);
        case Type::Softmax:
            return softmax(matrix);
        default:
            throw std::invalid_argument("unknown activation type");
    }
}

Matrix activation::apply_prime(const Matrix& matrix, activation::Type type) {
    switch (type) {
        case Type::ReLU:
            return matrix.apply(ReLU_prime);
        case Type::Sigmoid:
            return matrix.apply(sigmoid_prime);
        case Type::Softmax:
            throw std::logic_error("softmax derivative should be handled in loss calculation");
        default:
            throw std::invalid_argument("unknown activation type");
    }
}

double activation::ReLU(double val) {
    return val >= 0.0 ? val : 0.0;
}

double activation::ReLU_prime(double val) {
    return val >= 0.0 ? 1.0 : 0.0;
}

double activation::sigmoid(double val) {
    return 1.0 / (1.0 + std::exp(-val));
}

double activation::sigmoid_prime(double val) {
    const double sigmoid = 1.0 / (1.0 + std::exp(-val));
    return sigmoid * (1.0 - sigmoid);
}

Matrix activation::softmax(const Matrix& logits) {
    const int rows = logits.rows();
    const int cols = logits.cols();
    Matrix out { rows, cols };
    
    for (int col { 0 }; col < cols; ++col) {
        double col_max = logits[0, col];
        
        for (int row { 1 }; row < rows; ++row) {
            col_max = std::max(col_max, logits[row, col]);
        }
        
        double sum_exp = 0.0;
        for (int row { 0 }; row < rows; ++row) {
            double exp = std::exp(logits[row, col] - col_max);
            out[row, col] = exp;
            sum_exp += exp;
        }
        
        double inverse_sum = 1 / sum_exp;
        for (int row { 0 }; row < rows; ++row) {
            out[row, col] *= inverse_sum;
        }
    }
    return out;
}
