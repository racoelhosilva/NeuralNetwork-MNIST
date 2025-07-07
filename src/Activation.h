#pragma once

#include "Matrix.h"

/**
 * @namespace activation
 * @brief Contains functions and types related to activation functions.
 */
namespace activation {

    /**
     * @enum Type
     * @brief Enum representing the type of activation function.
     */
    enum class Type {
        ReLU,       ///< Rectified Linear Unit activation function
        Sigmoid,    ///< Sigmoid activation function
        Softmax     ///< Softmax activation function
    };

    /**
     * @brief Applies the specified activation function to a matrix.
     * @param matrix The input matrix.
     * @param type The type of activation function to apply.
     * @return A new matrix with the activation function applied.
     * @throws std::invalid_argument if an unknown activation type is specified.
     * @throws std::logic_error if the softmax function is applied and the sum of 
     * exponentials in any column is zero
     */
    [[nodiscard]] Matrix apply(const Matrix& matrix, activation::Type type);

    /**
     * @brief Applies the derivative of the specified activation function to a matrix.
     * @param matrix The input matrix.
     * @param type The type of activation function to apply the derivative of.
     * @return A new matrix with the derivative of the activation function applied.
     * @throws std::invalid_argument if an unknown activation type is specified.
     * @throws std::logic_error if the softmax function is applied, as its derivative
     * should be handled in the loss calculation.
     */
    [[nodiscard]] Matrix apply_prime(const Matrix& matrix, activation::Type type);

    /**
     * @brief Applies the ReLU activation function to a value.
     * @param val The input value.
     * @return The output value after applying ReLU.
     */
    [[nodiscard]] double ReLU(double val) noexcept;

    /**
     * @brief Applies the derivative of the ReLU activation function to a value.
     * @param val The input value.
     * @return The output value after applying the derivative of ReLU.
     */
    [[nodiscard]] double ReLU_prime(double val) noexcept;

    /**
     * @brief Applies the sigmoid activation function to a value.
     * @param val The input value.
     * @return The output value after applying the sigmoid function.
     */
    [[nodiscard]] double sigmoid(double val) noexcept;

    /**
     * @brief Applies the derivative of the sigmoid activation function to a value.
     * @param val The input value.
     * @return The output value after applying the derivative of the sigmoid function.
     */
    [[nodiscard]] double sigmoid_prime(double val) noexcept;
    
    /**
     * @brief Applies the softmax activation function to a matrix of logits.
     * @param logits The input matrix containing logits.
     * @return A new matrix with the softmax function applied.
     * @throws std::logic_error if the sum of exponentials in any column is zero,
     * which can occur if all logits in that column are the same and equal to negative infinity
     */
    [[nodiscard]] Matrix softmax(const Matrix& logits);
}
