#include "LearningRate.h"
#include <cmath>
#include <stdexcept>

double learning_rate::current(
    const learning_rate::Settings& lr,
    int epoch
) {
    if (epoch < 0) {
        throw std::invalid_argument("epoch must be non-negative");
    }
    switch (lr.type) {
        case Type::Constant:
            return lr.initial;
        case Type::Exponential:
            return lr.initial * std::exp(-lr.k * epoch);
        case Type::InvSqrt:
            return lr.initial / std::sqrt(epoch + 1);
        case Type::TimeBased:
            return lr.initial / (1.0 + lr.k * epoch);
        default:
            throw std::invalid_argument("unknown learning rate type");
    }
}
