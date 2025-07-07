#include "Layer.h"

Matrix Layer::predict(const Matrix& a_prev) const {
    return activation::apply(broadcast_col_add(w * a_prev, b), activation);
}

Matrix Layer::forward(const Matrix& a_prev) {
    cached_input = a_prev;
    z = broadcast_col_add(w * a_prev, b);
    return activation::apply(z, activation);
}

Matrix Layer::backward(const Matrix& gradient) {
    Matrix delta = gradient
        .hadamard(activation::apply_prime(z, activation));
    
    const int batch_size = cached_input.cols();
    dw = (delta * cached_input.transpose()) / batch_size;
    db = delta.row_avg();

    return w.transpose() * delta;
}

std::pair<Matrix, double> Layer::loss(const Matrix& label, const Matrix& prediction, loss::Type loss) {
    double loss_metric = loss::compute(label, prediction, loss);

    Matrix delta = loss::gradient(label, 
        prediction, 
        z,
        loss,
        activation
    );
    
    const double batch_size = cached_input.cols();
    dw = (delta * cached_input.transpose()) / batch_size;
    db = delta.row_avg();
    
    return {w.transpose() * delta, loss_metric};
}

void Layer::update(double learning_rate, 
    const regularization::Settings& regularization,
    double weight_decay
) {
    const Matrix reg_term = regularization::term(w, regularization);
    if (weight_decay > 0.0) {
        w -= learning_rate * weight_decay * w;
    }
    optimizer_w->update(w, dw + reg_term, learning_rate);
    optimizer_b->update(b, db, learning_rate);
}

Matrix Layer::broadcast_col_add(const Matrix& matrix, const Matrix& column) const {
    if (column.rows() != matrix.rows() || column.cols() != 1) {
        throw std::invalid_argument("broadcast column add input size mismatch");
    }
    Matrix result = matrix;
    for (int row = 0; row < result.rows(); ++row) {
        const double b = column[row, 0];
        for (int col = 0; col < result.cols(); ++col) {
            result[row, col] += b;
        }
    }

    return result;
}
