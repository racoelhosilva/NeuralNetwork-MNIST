#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include "Loss.h"
#include "Matrix.h"
#include "Regularization.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int input, int hidden, int output);

    void train(const Matrix& input, const Matrix& label, double learning_rate);

    void fit(const Matrix& input, const Matrix& label, int epochs, double learning_rate, int batch_size, const Matrix& val_input, const Matrix& val_label);

    [[nodiscard]] double evaluate(const Matrix& input, const Matrix& labels) const;

    [[nodiscard]] Matrix predict(const Matrix& input) const;
private:
    loss::Type loss = loss::Type::CrossEntropy;
    regularization::Type regularization = regularization::Type::Elastic;
    double lambda1 = 0.0001;
    double lambda2 = 0.0001;
    std::vector<Layer> layers;
};

#endif
