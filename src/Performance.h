#pragma once

#include <ostream>

/**
 * @namespace performance
 * @brief Contains performance metrics for model evaluation.
 */
namespace performance {

    /**
     * @struct metrics
     * @brief Holds performance metrics for model evaluation.
     */
    struct metrics {
        [[nodiscard]] double loss = 0.0;
        [[nodiscard]] double accuracy = 0.0;
    };
}

/**
 * @brief Outputs performance metrics to a stream.
 * @param out Output stream.
 * @param metrics Performance metrics to output.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream& out, const performance::metrics& metrics);
