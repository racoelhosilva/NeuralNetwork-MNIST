#ifndef CONFIG_H
#define CONFIG_H

#include "Activation.h"
#include "Initialization.h"
#include "LearningRate.h"
#include "Loss.h"
#include "Matrix.h"
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
        regularization::Type regularization_type;
        double lambda1;
        double lambda2;
    };

    struct Training {
        int epochs;
        int batch_size;
        learning_rate::Type learning_rate_type;
        double learning_rate;
        double k;
    };

    struct Validation {
        const Matrix& X;
        const Matrix& y;
        int patience;
    };
}

#endif
