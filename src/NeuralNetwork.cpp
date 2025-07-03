#include "NeuralNetwork.h"
#include <cmath>
#include <iostream>
#include <random>

static std::mt19937 generator(std::random_device{}());

NeuralNetwork::NeuralNetwork(int input, int hidden, int output) {
    layers.push_back({input, hidden, activation::Type::ReLU, initialization::Type::He, generator});
    layers.push_back({hidden, hidden, activation::Type::ReLU, initialization::Type::He, generator});
    layers.push_back({hidden, output, activation::Type::Softmax, initialization::Type::Glorot, generator});
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

void NeuralNetwork::fit(const Matrix& input, const Matrix& label, int epochs, double learning_rate, int batch_size) {
    const int num_samples = input.cols();

    for (int epoch { 0 }; epoch < epochs; ++epoch) {
        
        std::cout << "Epoch " << epoch << '\n';
    
        for (int start { 0 }; start < num_samples; start += batch_size) {
            int end = std::min(start + batch_size, num_samples);

            Matrix batch_inputs = input.cols(start, end);
            Matrix batch_labels = label.cols(start, end);
        
            train(batch_inputs, batch_labels, learning_rate);
        }
    }
}


Matrix NeuralNetwork::predict(const Matrix& input) const {
    Matrix a = input;
    for (auto& layer : layers) {
        a = layer.predict(a);
    }
    return a;
}
