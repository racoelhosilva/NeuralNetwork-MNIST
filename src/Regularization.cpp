#include "Regularization.h"
#include <stdexcept>

Matrix regularization::term(
    const Matrix& w, 
    const regularization::Settings& settings
) {
    auto sign = [](double v) noexcept { 
        return (v > 0.0) ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
    };
    switch (settings.type) {
        case Type::L1:
            return settings.lambda1 * w.apply(sign);
        case Type::L2:
            return settings.lambda2 * w;
        case Type::Elastic:
            return settings.lambda1 * w.apply(sign) 
                + settings.lambda2 * w;
        case Type::None:
            return Matrix(w.rows(), w.cols(), 0.0);
        default:
            throw std::invalid_argument("unknown regularization type");
    }
}
