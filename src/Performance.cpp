#include "Performance.h"

std::ostream& operator<<(std::ostream& out, const performance::metrics& metrics) {
    out << "Loss: " << metrics.loss 
        << " | "
        << "Accuracy: " << metrics.accuracy * 100.0;
    return out;
}
