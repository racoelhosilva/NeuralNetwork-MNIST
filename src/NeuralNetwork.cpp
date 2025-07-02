#include "NeuralNetwork.h"
#include "Activation.h"
#include "Initialization.h"
#include <cmath>
#include <iostream>
#include <random>

static std::mt19937 generator(std::random_device{}());

NeuralNetwork::NeuralNetwork(int input, int hidden, int output)
    : l1(input, hidden, initialization::Type::He, generator)
    , l2(hidden, output, initialization::Type::Glorot, generator)
    {}

void NeuralNetwork::train_step(const Matrix& input, const Matrix& label, double learning_rate) {
    
    /* Forward Propagation */

    Matrix z1 = l1.w * input + l1.b;
    Matrix a1 = z1.apply(activation::ReLU);
    Matrix z2 = l2.w * a1 + l2.b;
    Matrix a2 = activation::softmax(z2);

    /* Backward Propagation */

    Matrix dz2 = a2 - label;
    Matrix dw2 = (dz2 * a1.transpose()); // for single input
    Matrix db2 = dz2; // for single input
    Matrix dz1 = (l2.w.transpose() * dz2)
        .hadamard(z1.apply(activation::ReLU_prime));
    Matrix dw1 = (dz1 * input.transpose()); // for single input
    Matrix db1 = dz1; // for single input

    /* Weights and Biases Update */

    l1.w -= learning_rate * dw1;
    l1.b -= learning_rate * db1;
    l2.w -= learning_rate * dw2;
    l2.b -= learning_rate * db2;
}

Matrix NeuralNetwork::predict(const Matrix& input) const {
    Matrix z1 = l1.w * input + l1.b;
    Matrix a1 = z1.apply(activation::ReLU);
    Matrix z2 = l2.w * a1 + l2.b;
    Matrix a2 = activation::softmax(z2);
    return a2;
}
