#include "Performance.h"
#include <format>

std::ostream& operator<<(std::ostream& out, const performance::metrics& metrics) {
    out << std::format("Loss: {} | Accuracy: {}", metrics.loss, metrics.accuracy * 100.0);
    return out;
}
