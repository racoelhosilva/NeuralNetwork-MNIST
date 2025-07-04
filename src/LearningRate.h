#ifndef LEARNING_RATE_H
#define LEARNING_RATE_H

namespace learning_rate {

    enum class Type {
        Constant,
        Exponential,
        InvSqrt,
        TimeBased
    };

    double current(
        double initial, 
        learning_rate::Type type, 
        int epoch, 
        double k
    );
}

#endif
