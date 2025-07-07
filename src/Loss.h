#pragma once

#include "Activation.h"
#include "Matrix.h"

/**
 * @namespace loss
 * @brief Contains functions and types related to loss functions.
 */
namespace loss {

    /**
     * @brief A small constant to avoid division by zero or log of zero.
     * @note This value is used in loss calculations to prevent numerical instability.
     */
    inline constexpr double EPSILON = 1e-8;

    /**
     * @enum Type
     * @brief Enum representing the type of loss function.
     */
    enum class Type {
        CrossEntropy,   ///< Cross-entropy loss function
        MSE             ///< Mean Squared Error loss function
    };

    /**
     * @brief Computes the loss between the label and prediction matrices.
     * @param label The ground truth labels.
     * @param prediction The predicted values.
     * @param type The type of loss function to use.
     * @return The computed loss value.
     * @throws std::invalid_argument if the label and prediction matrices have different shapes.
     */
    [[nodiscard]] double compute(
        const Matrix& label,
        const Matrix& prediction,
        loss::Type type
    );

    /**
     * @brief Computes the gradient of the loss function with respect to the prediction.
     * @param label The ground truth labels.
     * @param prediction The predicted values.
     * @param z The input to the activation function (used for backpropagation).
     * @param type The type of loss function to use.
     * @param activation The type of activation function used in the network.
     * @return A matrix representing the gradient of the loss with respect to the prediction.
     * @throws std::invalid_argument if an unsupported activation function is used with cross-entropy loss.
     */
    [[nodiscard]] Matrix gradient(
        const Matrix &label,
        const Matrix &prediction, 
        const Matrix &z, 
        loss::Type type, 
        activation::Type activation
    );
}
