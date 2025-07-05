#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Config.h"
#include "Layer.h"
#include "Loss.h"
#include "Matrix.h"
#include "Regularization.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const config::Network& config);

    void train(
        const Matrix& input, 
        const Matrix& label, 
        double learning_rate
    );

    void fit(
        const Matrix& input, 
        const Matrix& label, 
        const config::Training& config, 
        std::optional<config::Validation> validation = std::nullopt
    );

    [[nodiscard]] double evaluate(
        const Matrix& input, 
        const Matrix& labels
    ) const;

    [[nodiscard]] Matrix predict(
        const Matrix& input
    ) const;
private:
    loss::Type loss;
    regularization::settings regularization;
    std::vector<Layer> layers;

    static Matrix random_cols(const Matrix& data, const std::vector<int>& idx, int start, int end);
};

#endif
