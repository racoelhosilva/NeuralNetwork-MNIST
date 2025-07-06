#include "Config.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <iostream>

int main() {
    /* Training */

    std::string train_images = "data/train-images-idx3-ubyte"; 
    std::string train_labels = "data/train-labels-idx1-ubyte";

    auto[train_X, train_y] = 
        mnist::load(train_images, train_labels, 1000);

    std::cout << "Train Dataset: " 
        << train_X.rows() << " x " << train_X.cols() 
        << " | "
        << train_y.rows() << " x " << train_y.cols()
        << '\n';
    
    /* Testing */

    std::string test_images = "data/t10k-images-idx3-ubyte"; 
    std::string test_labels = "data/t10k-labels-idx1-ubyte";

    auto[test_X, test_y] = 
        mnist::load(test_images, test_labels, 100);

    std::cout << "Test Dataset: " 
        << test_X.rows() << " x " << test_X.cols() 
        << " | "
        << test_y.rows() << " x " << test_y.cols()
        << '\n';

    /* Training and Testing Model */

    config::Network network_config {
        .input_size = 784,
        .layers = {
            {16, activation::Type::ReLU, initialization::Type::He},
            {16, activation::Type::Sigmoid, initialization::Type::Glorot},
            {10, activation::Type::Softmax, initialization::Type::Glorot},
        },
        .loss_type = loss::Type::CrossEntropy,
        .optimizer = {
            .type = optimizer::Type::SGD,
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
        .batch_size = 1,
        .shuffle = true,
        .learning_rate = {
            .initial = 0.01,
            .type = learning_rate::Type::TimeBased,
            .k = 0.05,
        },
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
