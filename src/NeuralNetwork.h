#pragma once

#include "Config.h"
#include "Layer.h"
#include "Loss.h"
#include "Matrix.h"
#include "Performance.h"
#include "Regularization.h"
#include <optional>
#include <vector>

/**
 * @class NeuralNetwork
 * @brief Represents a neural network model.
 */
class NeuralNetwork {
public:
    /**
     * @brief Constructs a NeuralNetwork instance from a configuration.
     * @param config The configuration for the neural network.
     */
    NeuralNetwork(const config::Network& config);

    /**
     * @brief Trains the neural network using the provided input and label matrices.
     * @param input The input data matrix.
     * @param label The label data matrix.
     * @param learning_rate The learning rate for the training process.
     * @note This function performs a single training step and should be called iteratively.
     */
    void train(
        const Matrix& input, 
        const Matrix& label, 
        double learning_rate
    );

    /**
     * @brief Fits the model to the training data.
     * @param input The input data matrix.
     * @param label The label data matrix.
     * @param config The training configuration.
     * @param validation Optional validation configuration for improvement and early stopping.
     */
    void fit(
        const Matrix& input, 
        const Matrix& label, 
        const config::Training& config, 
        std::optional<config::Validation> validation = std::nullopt
    );

    /**
     * @brief Evaluates the model's performance on the given input and labels.
     * @param input The input data matrix.
     * @param labels The label data matrix.
     * @param loss_type The type of loss function to use for evaluation.
     * @return A metrics object containing the loss and accuracy.
     */
    [[nodiscard]] performance::metrics evaluate(
        const Matrix& input, 
        const Matrix& labels,
        loss::Type loss_type
    ) const;

    /**
     * @brief Predicts the output for the given input data.
     * @param input The input data matrix.
     * @return The predicted output matrix.
     */
    [[nodiscard]] Matrix predict(
        const Matrix& input
    ) const;
private:

    /**
     * @brief Loss function type used in the neural network.
     */
    loss::Type loss;

    /**
     * @brief Regularization settings for the neural network.
     */
    regularization::Settings regularization;

    /**
     * @brief Weight decay factor for regularization.
     */
    double weight_decay;

    /**
     * @brief Layers of the neural network.
     */
    std::vector<Layer> layers;

    /**
     * @brief Loss value for the current epoch.
     */
    double epoch_loss;

    /**
     * @brief Random batch matrix generator from the input data with given order and range.
     * @param data The input data matrix.
     * @param idx The order of indices to select columns from the data.
     * @param start The starting index for the range of columns to select.
     * @param end The ending index for the range of columns to select.
     * @return A new matrix containing the selected columns from the input data.
     * @note This function is used to create batches of data for training.
     */
    static Matrix random_cols(const Matrix& data, const std::vector<int>& idx, int start, int end);
};
