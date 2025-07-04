#include "Config.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <iostream>

int main() {
    /* Training */

    std::string train_images = "data/train-images-idx3-ubyte"; 
    std::string train_labels = "data/train-labels-idx1-ubyte";

    auto[train_X, train_y] = 
        mnist::load(train_images, train_labels);

    std::cout << "Train Dataset: " 
        << train_X.rows() << " x " << train_X.cols() 
        << " | "
        << train_y.rows() << " x " << train_y.cols()
        << '\n';
    
    /* Testing */

    std::string test_images = "data/t10k-images-idx3-ubyte"; 
    std::string test_labels = "data/t10k-labels-idx1-ubyte";

    auto[test_X, test_y] = 
        mnist::load(test_images, test_labels);

    std::cout << "Test Dataset: " 
        << test_X.rows() << " x " << test_X.cols() 
        << " | "
        << test_y.rows() << " x " << test_y.cols()
        << '\n';

    /* Training and Testing Model */

    config::Network network_config;
    network_config.input_size = 784;
    network_config.layers = {
        {80, activation::Type::ReLU, initialization::Type::He},
        {80, activation::Type::Sigmoid, initialization::Type::Glorot},
        {10, activation::Type::Softmax, initialization::Type::Glorot},
    };
    network_config.loss_type = loss::Type::CrossEntropy;
    network_config.regularization_type = regularization::Type::L2;
    network_config.lambda1 = 0.0001;
    network_config.lambda2 = 0.0001; 

    NeuralNetwork model { network_config };

    config::Training training_config;
    training_config.epochs = 50;
    training_config.learning_rate = 0.01;
    training_config.batch_size = 16;

    config::Validation validation {
        test_X, 
        test_y
    };

    model.fit(train_X, train_y, training_config, validation);

    return 0;
}
