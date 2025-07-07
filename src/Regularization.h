#pragma once

#include "Matrix.h"

/**
 * @namespace regularization
 * @brief Contains functions and types related to regularization techniques.
 */
namespace regularization {

    /**
     * @enum Type
     * @brief Enum representing the type of regularization.
     */
    enum class Type {
        None,       ///< No regularization
        L1,         ///< L1 regularization (Lasso)
        L2,         ///< L2 regularization (Ridge)
        Elastic     ///< Elastic Net regularization (combination of L1 and L2)
    };

    /**
     * @struct Settings
     * @brief Struct containing settings for regularization.
     */
    struct Settings {
        regularization::Type type = Type::None; ///< Type of regularization
        double lambda1 = 0.0;                   ///< Coefficient for L1 regularization
        double lambda2 = 0.0;                   ///< Coefficient for L2 regularization
    };

    /**
     * @brief Computes the regularization term for a given matrix and settings.
     * @param w The matrix to apply regularization to.
     * @param settings The settings for the regularization.
     * @return A new Matrix representing the regularization term.
     * @throws std::invalid_argument if an unknown regularization type is specified.
     */
    [[nodiscard]] Matrix term(
        const Matrix& w, 
        const regularization::Settings& settings
    );
}
