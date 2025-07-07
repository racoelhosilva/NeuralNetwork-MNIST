#pragma once

#include "Activation.h"
#include "Initialization.h"
#include "Loss.h"
#include "Optimizer.h"
#include "Regularization.h"
#include <memory>
#include <random>

/**
 * @class Layer
 * @brief Represents a single layer in a neural network.
 */
class Layer {
public:
    /**
     * @brief Constructs a Layer object with the specified parameters.
     * @param input Number of input features.
     * @param output Number of output features.
     * @param activation Activation function type.
     * @param initialization Weight initialization type.
     * @param optimizer Optimizer settings for weight and bias updates.
     * @param gen Random number generator for weight initialization.
     */
    Layer(
        int input, int output, 
        activation::Type activation, 
        initialization::Type initialization,
        const optimizer::Settings& optimizer, 
        std::mt19937& gen
    )
        : activation {activation}
        , w(initialization::init(output, input, initialization, gen))
        , b(output, 1, 0.0)
        , z(output, 1)
        , cached_input(input, 1)
        , dw(output, input)
        , db(output, 1)
    {
        optimizer_w = optimizer::create(output, input, optimizer);
        optimizer_b = optimizer::create(output, 1, optimizer);
    }

    /**
     * @brief Forwards the input through the layer.
     * @param a_prev Input matrix from the previous layer.
     * @return Output matrix after applying the activation function.
     */
    [[nodiscard]] Matrix forward(const Matrix& a_prev);

    /**
     * @brief Backwards the gradient through the layer.
     * @param gradient Gradient matrix from the next layer.
     * @return Gradient matrix for the previous layer.
     */
    [[nodiscard]] Matrix backward(const Matrix& gradient);
    
    /**
     * @brief Computes the loss and its gradient for the layer.
     * @param label True labels for the input data.
     * @param prediction Predicted output from the layer.
     * @param loss Type of loss function to use.
     * @return Pair containing the gradient for the previous layer and the computed loss value.
     * @note This function should be used on the output layer.
     */
    [[nodiscard]] std::pair<Matrix, double> loss(const Matrix& label, const Matrix& prediction, loss::Type loss);

    /**
     * @brief Updates the weights and biases using the optimizer, learning rate, regularization, and weight decay settings.
     * @param learning_rate Learning rate for the update.
     * @param regularization Regularization settings for the update.
     * @param weight_decay Weight decay factor for regularization.
     */
    void update(double learning_rate, const regularization::Settings& regularization, double weight_decay);
    
    /**
     * @brief Predicts the output for the given input using the layer's weights and biases.
     * @param a_prev Input matrix to predict from.
     * @return Predicted output matrix after applying the activation function.
     */
    [[nodiscard]] Matrix predict(const Matrix& a_prev) const;
private:
    /**
     * @brief Activation function used in the layer.
     */
    activation::Type activation;

    /**
     * @brief Weight matrix for the layer.
     */
    Matrix w;

    /**
     * @brief Bias vector for the layer.
     */
    Matrix b;

    /**
     * @brief Linear combination of inputs and weights.
     */
    Matrix z;

    /**
     * @brief Cached input from the previous layer for backpropagation.
     */
    Matrix cached_input;

    /**
     * @brief Gradient of the weights with respect to the loss.
     */
    Matrix dw;

    /**
     * @brief Gradient of the biases with respect to the loss.
     */
    Matrix db;

    /**
     * @brief Optimizer for the weights.
     */
    std::shared_ptr<optimizer::Base> optimizer_w;
    
    /**
     * @brief Optimizer for the biases.
     */
    std::shared_ptr<optimizer::Base> optimizer_b;

    /**
     * @brief Adds a column vector to each column of a matrix.
     * @param matrix The matrix to which the column vector will be added.
     * @param column The column vector to be added.
     * @return A new matrix with the column vector added to each column of the input matrix.
     * @throws std::invalid_argument if the column vector's size does not match the number
     * of rows in the matrix or if the column vector is not a single column.
     */
    [[nodiscard]] Matrix broadcast_col_add(const Matrix& matrix, const Matrix& column) const;
};
