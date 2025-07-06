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
        int units;
        activation::Type activation_type;
        initialization::Type initialization_type;
    };

    struct Network {
        int input_size;
        std::vector<config::Layer> layers;
        loss::Type loss_type;
        double weight_decay = 0.0;
        optimizer::settings optimizer;
        regularization::settings regularization;
    };

    struct Training {
        int epochs;
        int batch_size;
        bool shuffle;
        learning_rate::settings learning_rate;
    };

    struct Validation {
        Matrix& X;
        Matrix& y;
        bool early_stop;
        int patience = 0;
    };
}

#endif
