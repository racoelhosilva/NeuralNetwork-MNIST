#ifndef LAYER_H
#define LAYER_H

#include "Initialization.h"
#include <random>

class Layer {
public:
    Layer(int input, int output, initialization::Type type, std::mt19937& gen)
        : w(initialization::init(output, input, type, gen))
        , b(Matrix(output, 1, 0.0))
    {}

    Matrix w;
    Matrix b;
};

#endif
