#ifndef LEARNING_RATE_H
#define LEARNING_RATE_H

namespace learning_rate {

    enum class Type {
        Constant,
        Exponential,
        InvSqrt,
        TimeBased
    };

    struct settings {
        learning_rate::Type type = Type::Constant;
        double initial = 0.001;
        double k = 0.0;
    };

    double current(
        const learning_rate::settings& learning_rate,
        int epoch
    );
}

#endif
