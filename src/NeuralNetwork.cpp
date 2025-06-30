#include "NeuralNetwork.h"
#include <math.h>
#include <iostream>
#include <random>

static std::mt19937 generator({std::random_device{}()});

NeuralNetwork::NeuralNetwork(int input, int hidden, int output)
    : m_input { input }
    , m_hidden { hidden }
    , m_output { output }
    , w1(Matrix::random(
        hidden, input, 
        std::normal_distribution(0.0, std::sqrt(2.0 / input)), 
        generator))
    , b1(Matrix {hidden, 1, 0.0})
    , w2(Matrix::random(
        output, hidden, 
        std::normal_distribution(0.0, std::sqrt(2.0 / hidden)),
        generator))
    , b2(Matrix {output, 1, 0.0})
    {}

