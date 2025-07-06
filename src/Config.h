#ifndef CONFIG_H
#define CONFIG_H

#include "Activation.h"
#include "Initialization.h"
#include "LearningRate.h"
#include "Loss.h"
#include "Matrix.h"
#include "Optimizer.h"
#include "Regularization.h"
#include <vector>

namespace config {

    struct Layer {
        int units = 1;
        activation::Type activation_type = activation::Type::Sigmoid;
        initialization::Type initialization_type = initialization::Type::He;
    };

    struct Network {
        int input_size = 0;
        std::vector<config::Layer> layers;
        loss::Type loss_type = loss::Type::CrossEntropy;
        double weight_decay = 0.0;
        optimizer::settings optimizer;
        regularization::settings regularization;
    };

    struct Training {
        int epochs = 20;
        int batch_size = 32;
        bool shuffle = true;
        learning_rate::settings learning_rate;
        bool best_model = true;
    };

    struct Validation {
        Matrix& X;
        Matrix& y;
        bool early_stop = true;
        int patience = 5;
    };
}

#endif
