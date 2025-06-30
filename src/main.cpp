#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <iostream>

Matrix label_to_matrix(int label) {
    Matrix res(10, 1, 0.0);
    res.at(label, 0) = 1.0;
    return res;
}

int main() {
    /* Training */

    std::string train_images = "data/train-images-idx3-ubyte"; 
    std::string train_labels = "data/train-labels-idx1-ubyte";

    std::vector<mnist::Record> train = 
        mnist::load(train_images, train_labels, 1000);

    std::cout << train.size() << '\n';
    
    NeuralNetwork model { 784, 80, 10 };

    for (const auto& r : train) {
        model.train(r.input.flatten(), label_to_matrix(r.label), 0.1);
    }

    /* Testing */

    // std::string test_images = "data/t10k-images-idx3-ubyte"; 
    // std::string test_labels = "data/t10k-labels-idx1-ubyte";

    // std::vector<mnist::Record> test = 
    //     mnist::load(test_images, test_labels);

    // std::cout << test.size() << '\n';

    return 0;
}
