#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <ostream>

namespace performance {

    struct metrics {
        double loss;
        double accuracy;
    };
}

std::ostream& operator<<(std::ostream& out, const performance::metrics& metrics);


#endif
