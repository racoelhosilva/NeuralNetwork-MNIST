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

void NeuralNetwork::train_step(const Matrix& input, const Matrix& label, double learning_rate) {
    Matrix a = input;
    for (auto& layer : layers) {
        a = std::move(layer.forward(a));
    }

    Matrix dz = layers.back().loss(label, a, loss);
    for (int i = layers.size() - 2; i >= 0; i--) {
        dz = std::move(layers[i].backward(dz));
    }

    for (auto& layer : layers) {
        layer.update(learning_rate);
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) const {
    Matrix a = input;
    for (auto& layer : layers) {
        a = layer.predict(a);
    }
    return a;
}
