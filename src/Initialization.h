#pragma once

#include "Matrix.h"
#include <random>

/**
 * @namespace initialization
 * @brief Contains functions and types related to weight initialization techniques.
 */
namespace initialization {
    
    /**
     * @enum Type
     * @brief Enum representing the type of weight initialization.
     */
    enum class Type {
        LeCun,      ///< LeCun initialization (uniform distribution)
        Glorot,     ///< Glorot/Xavier initialization (uniform distribution)
        He          ///< He/Kaiming initialization (normal distribution)
    };

    /**
     * @brief Initializes a matrix with random values based on the specified type.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param type Type of initialization to use.
     * @param gen Random number generator to use for generating random values.
     * @return A new Matrix initialized with random values.
     * @throws std::invalid_argument if an unknown initialization type is specified.
     */
    [[nodiscard]] Matrix init(
        int rows, int cols, 
        initialization::Type type, 
        std::mt19937& gen
    );

    /**
     * @namespace detail
     * @brief Contains implementation details for the initialization functions.
     */
    namespace detail {
        /**
         * @brief Generates a matrix with random values based on the specified distribution.
         * @param rows Number of rows in the matrix.
         * @param cols Number of columns in the matrix.
         * @param dist Distribution to use for generating random values.
         * @param gen Random number generator to use for generating random values.
         * @return A new Matrix initialized with random values.
         */
        template <typename Distribution>
        [[nodiscard]] Matrix random(int rows, int cols, 
            Distribution&& dist, 
            std::mt19937 &gen
        );
    }
}
