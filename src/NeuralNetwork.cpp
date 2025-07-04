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
        layers.push_back({input, l.units, l.activation_type, l.initialization_type, generator});
        input = l.units;
    }
    loss = config.loss_type;
    regularization = config.regularization_type;
    lambda1 = config.lambda1;
    lambda2 = config.lambda2;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& label, double learning_rate) {
    Matrix a = input;
    for (auto& layer : layers) {
        a = std::move(layer.forward(a));
    }

    Matrix dz = layers.back().loss(label, a, loss);
    for (int i = layers.size() - 2; i >= 0; i--) {
        dz = std::move(layers[i].backward(dz));
    }

    for (auto& layer : layers) {
        layer.update(learning_rate, regularization, lambda1, lambda2);
    }
}

void NeuralNetwork::fit(
        const Matrix& input, 
        const Matrix& label, 
        const config::Training& config, 
        std::optional<config::Validation> validation
    ) {
    const int num_samples = input.cols();

    std::unique_ptr<NeuralNetwork> best_model;
    double best_accuracy = std::numeric_limits<double>::lowest();
    int patience = 0;

    std::vector<int> order(num_samples);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch { 0 }; epoch < config.epochs; ++epoch) {
        
        std::cout << "Epoch " << epoch+1 << " / " << config.epochs << '\n';
    
        if (config.shuffle) {
            std::shuffle(order.begin(), order.end(), generator);
        }

        const double learning_rate = learning_rate::current(
            config.learning_rate, 
            config.learning_rate_type, 
            epoch, 
            config.k
        );

        for (int start { 0 }; start < num_samples; start += config.batch_size) {
            int end = std::min(start + config.batch_size, num_samples);

            Matrix batch_inputs = NeuralNetwork::random_cols(input, order, start, end);
            Matrix batch_labels = NeuralNetwork::random_cols(label, order, start, end);

            train(batch_inputs, batch_labels, learning_rate);
        }

        if (validation.has_value()) {
            double accuracy = evaluate(validation.value().X, validation.value().y);
            
            std::cout << "Accuracy: " 
                << accuracy * 100.0 
                << '\n';
            
            if (!best_model || accuracy > best_accuracy) {
                best_model = std::make_unique<NeuralNetwork>(*this);
                best_accuracy = accuracy;
                patience = 0;
            } else if (validation.value().patience 
                && ++patience >= validation.value().patience) {
                std::cout << "Early stop triggered" << '\n';
                *this = *best_model;
                break;
            }
        }
    }

    if (validation.has_value()) {
        std::cout << "Final Accuracy: " 
        << evaluate(validation.value().X, validation.value().y) * 100.0 
        << '\n';
    }
}

double NeuralNetwork::evaluate(const Matrix& input, const Matrix& labels) const {
    Matrix pred = predict(input);
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
    return correct / labels.cols();
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

