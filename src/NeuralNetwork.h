#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input, int hidden, int output);

    void train(const Matrix& input, Matrix label, double learning_rate);
private:
    int m_input;
    int m_hidden;
    int m_output;

    Matrix w1;
    Matrix b1;
    Matrix w2;
    Matrix b2;
};

#endif
