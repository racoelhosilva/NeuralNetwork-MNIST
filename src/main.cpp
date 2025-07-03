#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <iostream>

int argmax(const Matrix& m) {
    int res = 0;
    for (int i { 1 }; i < 10; ++i) {
        if (m.at(i,0) > m.at(res,0)) {
            res = i;
        }
    }
    return res;
}

void accuracy(const NeuralNetwork& model, const Matrix& input, const Matrix& label) {
    double correct = 0;
    for (int idx { 0 }; idx < input.cols(); ++idx) {
        if (argmax(model.predict(input.col(idx))) == argmax(label.col(idx))) {
            correct += 1;
        }
    }
    std::cout << "Correct " << correct << " Accuracy " << (100.0 * correct) / input.cols() << '\n';
}

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

    for (int iter = 1; iter <= 50; ++iter) {
        for (int idx { 0 }; idx < train_X.cols(); ++idx) {
            model.train_step(train_X.col(idx), train_y.col(idx), 0.01);
        }

        std::cout << " > Iteration " << iter << "\n";
        accuracy(model, test_X, test_y);
    }

    return 0;
}
