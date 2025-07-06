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
        double initial;
        learning_rate::Type type;
        double k;
    };

    double current(
        const learning_rate::settings& learning_rate,
        int epoch
    );
}

#endif
