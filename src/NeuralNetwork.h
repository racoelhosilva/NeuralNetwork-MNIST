#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Matrix.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input, int hidden, int output);

    void train_step(const Matrix& input, const Matrix& label, double learning_rate);

    [[nodiscard]] Matrix predict(const Matrix& input) const;
private:
    const int m_input;
    const int m_hidden;
    const int m_output;
    
    Matrix w1;
    Matrix b1;
    Matrix w2;
    Matrix b2;
};

#endif
