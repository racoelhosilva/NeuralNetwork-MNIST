#include "Initialization.h"
#include <stdexcept>

namespace initialization {
    Matrix init(int rows, int cols, initialization::Type type, std::mt19937& gen) {
        const double fan_in  = static_cast<double>(cols);
        const double fan_out = static_cast<double>(rows);
        
        switch (type) {
            case Type::LeCun: {
                const double limit = std::sqrt(3.0 / fan_in);
                return detail::random(
                    rows, cols, 
                    std::uniform_real_distribution(-limit, limit),
                    gen
                );
            }
            case Type::Glorot: {
                const double limit = std::sqrt(6.0 / (fan_in + fan_out));
                return detail::random(
                    rows, cols, 
                    std::uniform_real_distribution(-limit, limit),
                    gen
                );
            }
            case Type::He: {
                const double stddev = std::sqrt(2.0 / fan_in);
                return detail::random(
                    rows, cols, 
                    std::normal_distribution(0.0, stddev),
                    gen
                );
            }
            default: 
                throw std::invalid_argument("unknown initialization type");
        }
    }

    namespace detail {
        template <typename Distribution>
        Matrix random(int rows, int cols, Distribution&& dist, std::mt19937 &gen) {
            std::vector<double> data(rows * cols);

            for (auto& elem : data) {
                elem = dist(gen);
            }

            return Matrix(rows, cols, std::move(data));
        }
    }
}