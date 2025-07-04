#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
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

void NeuralNetwork::fit(const Matrix& input, const Matrix& label, int epochs, double learning_rate, int batch_size, const Matrix& val_input, const Matrix& val_label) {
    const int num_samples = input.cols();

    for (int epoch { 0 }; epoch < epochs; ++epoch) {
        
        std::cout << "Epoch " << epoch << '\n';
        std::cout << "Accuracy: " 
            << evaluate(val_input, val_label) * 100.0 
            << '\n';
    
        for (int start { 0 }; start < num_samples; start += batch_size) {
            int end = std::min(start + batch_size, num_samples);

            Matrix batch_inputs = input.cols(start, end);
            Matrix batch_labels = label.cols(start, end);
        
            train(batch_inputs, batch_labels, learning_rate);
        }
    }

    std::cout << "Final Accuracy: " 
        << evaluate(val_input, val_label) * 100.0 
        << '\n';
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
