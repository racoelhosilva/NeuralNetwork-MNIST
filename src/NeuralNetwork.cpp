#include "Activation.h"
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
        std::normal_distribution(0.0, std::sqrt(2.0 / (hidden + output))),
        generator))
    , b2(Matrix {output, 1, 0.0})
    {}

void NeuralNetwork::train(const Matrix& input, Matrix label, double learning_rate) {
    
    /* Forward Propagation */

    Matrix z1 = w1 * input + b1;
    Matrix a1 = z1.apply(activation::ReLU);
    Matrix z2 = w2 * a1 + b2;
    Matrix a2 = activation::softmax(z2);

    /* Backward Propagation */

    Matrix dz2 = a2 - label;
    Matrix dw2 = (dz2 * a1.transpose()); // for single input
    Matrix db2 = dz2; // for single input
    Matrix dz1 = (w2.transpose() * dz2)
        .elem_mult(z1.apply(activation::RelU_prime));
    Matrix dw1 = (dz1 * input.transpose()); // for single input
    Matrix db1 = dz1; // for single input

    /* Weights and Biases Update */

    w1 -= learning_rate * dw1;
    b1 -= learning_rate * db1;
    w2 -= learning_rate * dw2;
    b2 -= learning_rate * db2;
}

Matrix NeuralNetwork::predict(const Matrix& input) const {
    Matrix z1 = w1 * input + b1;
    Matrix a1 = z1.apply(activation::ReLU);
    Matrix z2 = w2 * a1 + b2;
    Matrix a2 = activation::softmax(z2);
    return a2;
}
