#pragma once

#include <ostream>

namespace performance {

    struct metrics {
        double loss;
        double accuracy;
    };
}

std::ostream& operator<<(std::ostream& out, const performance::metrics& metrics);
