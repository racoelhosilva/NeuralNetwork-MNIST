#include "Config.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <format>
#include <iostream>

int main() {
    /* Training Dataset */

    constexpr std::string_view train_images = "data/train-images-idx3-ubyte"; 
    constexpr std::string_view train_labels = "data/train-labels-idx1-ubyte";

    auto [train_X, train_y] = 
        mnist::load(train_images, train_labels, 1000);

    std::cout << std::format(
        "Train Dataset: {} x {} | {} x {}\n",
        train_X.rows(), train_X.cols(), train_y.rows(), train_y.cols()
    );
    
    /* Testing Dataset */

    constexpr std::string_view test_images = "data/t10k-images-idx3-ubyte"; 
    constexpr std::string_view test_labels = "data/t10k-labels-idx1-ubyte";

    auto [test_X, test_y] = 
        mnist::load(test_images, test_labels, 100);

    std::cout << std::format(
        "Test Dataset: {} x {} | {} x {}\n",
        test_X.rows(), test_X.cols(), test_y.rows(), test_y.cols()
    );

    /* Model Configuration and Training */

    config::Network network_config {
        .input_size = 784,
        .layers = {
            {64, activation::Type::ReLU, initialization::Type::He},
            {64, activation::Type::ReLU, initialization::Type::He},
            {10, activation::Type::Softmax, initialization::Type::Glorot},
        },
        .loss_type = loss::Type::CrossEntropy,
        .weight_decay = 0.0005,
        .optimizer = {
            .type = optimizer::Type::Adam,
            .beta1 = 0.9,
            .beta2 = 0.999,
        },
        .regularization = {
            .type = regularization::Type::L2,
            .lambda1 = 0.0001,
            .lambda2 = 0.0001,
        },
    };

    NeuralNetwork model { network_config };

    config::Training training_config {
        .epochs = 100,
        .batch_size = 64,
        .shuffle = true,
        .learning_rate = {
            .type = learning_rate::Type::TimeBased,
            .initial = 0.001,
            .k = 0.05,
        },
        .best_model = true,
    };

    config::Validation validation { 
        .X = test_X, 
        .y = test_y,
        .early_stop = true,
        .patience = 20,
    };

    model.fit(train_X, train_y, training_config, validation);

    return 0;
}
