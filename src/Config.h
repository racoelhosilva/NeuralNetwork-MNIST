#ifndef CONFIG_H
#define CONFIG_H

#include "Activation.h"
#include "Initialization.h"
#include "Loss.h"
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
}

#endif
