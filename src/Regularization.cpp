#include "Regularization.h"

Matrix regularization::term(
    const Matrix& w, 
    regularization::Type type, 
    double lambda1,
    double lambda2
) {
    auto sign = [](double v) noexcept { 
        return (v > 0.0) ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
    };
    switch (type) {
        case Type::L1:
            return lambda1 * w.apply(sign);
        case Type::L2:
            return lambda2 * w;
        case Type::Elastic:
            return lambda1 * w.apply(sign) + lambda2 * w;
        case Type::None:
            return Matrix(w.rows(), w.cols(), 0.0);
        default:
            throw std::invalid_argument("unknown regularization type");
    }
}
