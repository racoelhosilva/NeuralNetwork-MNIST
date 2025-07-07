#pragma once

/**
 * @namespace learning_rate
 * @brief Contains functions and types related to learning rate scheduling.
 */
namespace learning_rate {

    /**
     * @enum Type
     * @brief Enum representing the type of learning rate scheduling.
     */
    enum class Type {
        Constant,       ///< Constant learning rate
        Exponential,    ///< Exponential decay learning rate
        InvSqrt,        ///< Inverse square root decay learning rate
        TimeBased       ///< Time-based decay learning rate
    };

    /**
     * @struct Settings
     * @brief Struct containing settings for learning rate scheduling.
     */
    struct Settings {
        learning_rate::Type type = Type::Constant;  ///< Type of learning rate scheduling
        double initial = 0.001;                     ///< Initial learning rate
        double k = 0.0;                             ///< Decay factor
    };

    /**
     * @brief Get the current learning rate for a specific epoch.
     * @param settings The learning rate settings.
     * @param epoch The current epoch.
     * @return The current learning rate.
     * @throws std::invalid_argument if the epoch is negative 
     * or if an unknown learning rate type is specified.
     */
    [[nodiscard]] double current(
        const learning_rate::Settings& settings,
        int epoch
    );
}
