#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include "Loss.h"
#include "Matrix.h"

class NeuralNetwork {
public:
    NeuralNetwork(int input, int hidden, int output);

    void train_step(const Matrix& input, const Matrix& label, double learning_rate);

    [[nodiscard]] Matrix predict(const Matrix& input) const;
private:
    loss::Type loss = loss::Type::CrossEntropy;
    Layer l1;
    Layer l2;
};

#endif
