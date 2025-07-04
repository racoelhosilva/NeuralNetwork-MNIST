#include "LearningRate.h"
#include <cmath>
#include <stdexcept>

double learning_rate::current(
    double initial, 
    learning_rate::Type type, 
    int epoch, 
    double k
    ) {
    switch (type) {
        case Type::Constant:
            return initial;
        case Type::Exponential:
            return initial * std::exp(-k * epoch);
        case Type::InvSqrt:
            return initial / std::sqrt(epoch + 1);
        case Type::TimeBased:
            return initial / (1.0 + k * epoch);
        default:
            throw std::invalid_argument("unknown learning rate type");
    }
}
