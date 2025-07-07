#pragma once

#include "Matrix.h"
#include <optional>
#include <random>

namespace initialization {

    enum class Type {
        LeCun,
        Glorot,
        He
    };

    Matrix init(int rows, int cols, 
        initialization::Type type, 
        std::mt19937& gen
    );

    namespace detail {
        template <typename Distribution>
        Matrix random(int rows, int cols, 
            Distribution&& dist, 
            std::mt19937 &gen
        );
    }
}
