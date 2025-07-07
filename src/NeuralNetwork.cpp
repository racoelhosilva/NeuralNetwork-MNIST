#include "NeuralNetwork.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

static std::mt19937 generator(std::random_device{}());

NeuralNetwork::NeuralNetwork(const config::Network& config) {
    int input = config.input_size;
    for (const auto& l : config.layers) {
        layers.push_back({input, l.units, l.activation_type, l.initialization_type, config.optimizer, generator});
        input = l.units;
    }
    loss = config.loss_type;
    regularization = config.regularization;
    weight_decay = config.weight_decay;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& label, double learning_rate) {
    Matrix a = input;
    for (auto& layer : layers) {
        a = std::move(layer.forward(a));
    }

    auto[dz, batch_loss] = layers.back().loss(label, a, loss);
    epoch_loss += batch_loss;
    for (int i = layers.size() - 2; i >= 0; i--) {
        dz = std::move(layers[i].backward(dz));
    }

    for (auto& layer : layers) {
        layer.update(learning_rate, regularization, weight_decay);
    }
}

void NeuralNetwork::fit(
    const Matrix& input, 
    const Matrix& label, 
    const config::Training& config, 
    std::optional<config::Validation> validation
) {
    const int num_samples = input.cols();
    const int BATCH_PER_EPOCH = (num_samples + config.batch_size - 1) / config.batch_size;

    std::unique_ptr<NeuralNetwork> best_model;
    double best_accuracy = std::numeric_limits<double>::lowest();
    int patience = 0;

    std::vector<int> order(num_samples);
    std::iota(order.begin(), order.end(), 0);


    for (int epoch { 0 }; epoch < config.epochs; ++epoch) {
        
        std::cout << "Epoch " << epoch+1 << " / " << config.epochs << '\n';
        epoch_loss = 0.0;        

        const double learning_rate = learning_rate::current(config.learning_rate, epoch);

        if (config.shuffle) {
            std::shuffle(order.begin(), order.end(), generator);
        }

        for (int start { 0 }; start < num_samples; start += config.batch_size) {
            int end = std::min(start + config.batch_size, num_samples);

            Matrix batch_inputs = NeuralNetwork::random_cols(input, order, start, end);
            Matrix batch_labels = NeuralNetwork::random_cols(label, order, start, end);

            train(batch_inputs, batch_labels, learning_rate);
        }

        std::cout << "Epoch loss: " << epoch_loss / BATCH_PER_EPOCH << '\n';

        if (validation.has_value()) {
            performance::metrics metrics = 
                evaluate(validation.value().X, validation.value().y, loss);
            
            std::cout << "Validation " << metrics << '\n';
            
            if (!best_model || metrics.accuracy > best_accuracy) {
                best_model = std::make_unique<NeuralNetwork>(*this);
                best_accuracy = metrics.accuracy;
                patience = 0;
            } else if (validation.value().early_stop
                && ++patience >= validation.value().patience) {
                std::cout << "Early stop triggered." << '\n';
                break;
            }
        }
    }

    if (config.best_model) {
        std::cout << "Restoring best model." << '\n';
        *this = *best_model;
    }

    if (validation.has_value()) {
        std::cout << evaluate(validation.value().X, validation.value().y, loss) << '\n';
    }
}

performance::metrics NeuralNetwork::evaluate(const Matrix& input, const Matrix& labels, loss::Type loss_type) const {
    Matrix pred = predict(input);
    const double loss = loss::compute(labels, pred, loss_type);
    double correct = 0;
    for (int col { 0 }; col < labels.cols(); ++col) {
        int prediction = 0;
        int label = 0;
        for (int row { 1 }; row < labels.rows(); ++row) {
            if (pred[row, col] > pred[prediction, col]) {
                prediction = row;
            }
            if (labels[row, col] > labels[label, col]) {
                label = row;
            }
        }
        if (prediction == label) {
            correct += 1.0;
        }
    }
    return {loss, correct / labels.cols()};
}


Matrix NeuralNetwork::predict(const Matrix& input) const {
    Matrix a = input;
    for (auto& layer : layers) {
        a = layer.predict(a);
    }
    return a;
}

Matrix NeuralNetwork::random_cols(const Matrix& data, const std::vector<int>& idx, int start, int end) {
    int rows = data.rows();
    int cols = end - start;
    Matrix out(rows, cols);

    for (int j { 0 }; j < cols; ++j) {
        int src_col = idx[start + j];
        for (int r { 0 }; r < rows; ++r)
            out[r, j] = data[r, src_col];
    }
    return out;
}

