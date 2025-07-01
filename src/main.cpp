#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <iostream>

Matrix label_to_matrix(int label) {
    Matrix res(10, 1, 0.0);
    res.at(label, 0) = 1.0;
    return res;
}

int matrix_to_label(const Matrix& matrix) {
    int res = 0;
    for (int i { 1 }; i < 10; ++i) {
        if (matrix.at(i,0) > matrix.at(res,0)) {
            res = i;
        }
    }
    return res;
}

void accuracy(const NeuralNetwork& model, const std::vector<mnist::Record>& records) {
    double correct = 0;
    for (const auto& r : records) {
        if (matrix_to_label(model.predict(r.input.flatten())) == r.label) {
            correct += 1;
        }
    }
    std::cout << "Correct " << correct << " Accuracy " << (100.0 * correct) / records.size() << '\n';
}

int main() {
    /* Training */

    std::string train_images = "data/train-images-idx3-ubyte"; 
    std::string train_labels = "data/train-labels-idx1-ubyte";

    std::vector<mnist::Record> train = 
        mnist::load(train_images, train_labels);

    std::cout << "Train Dataset size: " << train.size() << '\n';

    /* Testing */

    std::string test_images = "data/t10k-images-idx3-ubyte"; 
    std::string test_labels = "data/t10k-labels-idx1-ubyte";

    std::vector<mnist::Record> test = 
        mnist::load(test_images, test_labels);

    std::cout << "Test Dataset size: " << test.size() << '\n';

    /* Training and Testing Model */

    NeuralNetwork model { 784, 80, 10 };

    for (int iter = 1; iter <= 100; ++iter) {
        for (const auto& r : train) {
            model.train_step(r.input.flatten(), label_to_matrix(r.label), 0.01);
        }

        std::cout << " > Iteration " << iter << "\n";
        accuracy(model, test);
    }

    return 0;
}
