#ifndef LAYER_H
#define LAYER_H

#include "Activation.h"
#include "Initialization.h"
#include "Loss.h"
#include <random>

class Layer {
public:
    Layer(int input, int output, 
        activation::Type activation, 
        initialization::Type initialization, 
        std::mt19937& gen
    )
        : activation {activation}
        , w(initialization::init(output, input, initialization, gen))
        , b(output, 1, 0.0)
        , z(output, 1)
        , cached_input(input, 1)
        , dw(output, input)
        , db(output, 1)
    {}

    Matrix forward(const Matrix& a_prev);
    Matrix backward(const Matrix& gradient);
    Matrix loss(const Matrix& label, const Matrix& prediction, loss::Type loss);
    void update(double learning_rate);
    Matrix predict(const Matrix& a_prev) const;

    activation::Type activation;
    Matrix w;
    Matrix b;

    Matrix z;
    Matrix cached_input;
    Matrix dw, db;
};

#endif
