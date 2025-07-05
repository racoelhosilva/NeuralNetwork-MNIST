#include "Regularization.h"

Matrix regularization::term(
    const Matrix& w, 
    regularization::settings reg
) {
    auto sign = [](double v) noexcept { 
        return (v > 0.0) ? 1.0 : (v < 0.0 ? -1.0 : 0.0);
    };
    switch (reg.type) {
        case Type::L1:
            return reg.lambda1 * w.apply(sign);
        case Type::L2:
            return reg.lambda2 * w;
        case Type::Elastic:
            return reg.lambda1 * w.apply(sign) 
                + reg.lambda2 * w;
        case Type::None:
            return Matrix(w.rows(), w.cols(), 0.0);
        default:
            throw std::invalid_argument("unknown regularization type");
    }
}
