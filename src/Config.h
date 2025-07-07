#pragma once

#include "Activation.h"
#include "Initialization.h"
#include "LearningRate.h"
#include "Loss.h"
#include "Matrix.h"
#include "Optimizer.h"
#include "Regularization.h"
#include <vector>

/**
 * @namespace config
 * @brief Contains configuration structures for neural network layers, networks, training, and validation.
 */
namespace config {

    /**
     * @struct Layer
     * @brief Represents a single layer in a neural network.
     * @details Contains the number of units, activation function type, and weight initialization type.
     */
    struct Layer {
        int units = 1;                                                              ///< Number of units in the layer (default = 1)
        activation::Type activation_type = activation::Type::Sigmoid;               ///< Activation function type (default = Sigmoid)
        initialization::Type initialization_type = initialization::Type::Glorot;    ///< Weight initialization type (default = Glorot)
    };

    /**
     * @struct Network
     * @brief Represents the configuration of a neural network.
     * @details Contains input size, layers, loss type, weight decay, optimizer settings,
     *          and regularization settings.
     */
    struct Network {
        int input_size = 0;                                 ///< Input size of the network (default = 0)
        std::vector<config::Layer> layers;                  ///< List of layers in the network
        loss::Type loss_type = loss::Type::CrossEntropy;    ///< Loss function type (default = CrossEntropy)
        double weight_decay = 0.0;                          ///< Weight decay (default = 0.0)
        optimizer::Settings optimizer;                      ///< Optimizer settings (default = SGD)
        regularization::Settings regularization;            ///< Regularization settings (default = None)
    };

    /**
     * @struct Training
     * @brief Represents the configuration for training a neural network.
     * @details Contains the number of epochs, batch size, shuffle flag, learning rate settings,
     *          and whether to save the best model.
     */
    struct Training {
        int epochs = 20;                        ///< Number of epochs for training (default = 20)
        int batch_size = 32;                    ///< Batch size for training (default = 32)
        bool shuffle = true;                    ///< Shuffle training data (default = true)
        learning_rate::Settings learning_rate;  ///< Learning rate settings (default = constant = 0.001)
        bool best_model = true;                 ///< Save the best model during training (default = true)
    };

    /**
     * @struct Validation
     * @brief Represents the configuration for validating a neural network.
     * @details Contains validation data, early stopping flag, and patience for early stopping.
     */
    struct Validation {
        Matrix& X;                  ///< Validation input data
        Matrix& y;                  ///< Validation target data
        bool early_stop = true;     ///< Enable early stopping (default = true)
        int patience = 5;           ///< Patience for early stopping (default = 5)
    };
}
