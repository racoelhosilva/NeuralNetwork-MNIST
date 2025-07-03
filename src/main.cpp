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

    NeuralNetwork model { 784, 16, 10 };

    return 0;
}
