#ifndef LAYER_H
#define LAYER_H

#include "Activation.h"
#include "Initialization.h"
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
        , b(Matrix(output, 1, 0.0))
    {}

    activation::Type activation;
    Matrix w;
    Matrix b;
};

#endif
