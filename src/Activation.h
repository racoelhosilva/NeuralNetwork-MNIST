#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"
#include <math.h>

namespace activation {
    double ReLU(double val) {
        return val >= 0.0 ? val : 0.0;
    }

    double RelU_prime(double val) {
        return val >= 0.0 ? val : 0.0;
    }

    Matrix softmax(const Matrix& logits) {
        const int R = logits.rows();
        const int C = logits.cols();
        Matrix out(R, C);

        for (int c = 0; c < C; ++c) {
            double col_max = logits[0, c];
            for (int r = 1; r < R; ++r)
                col_max = std::max(col_max, logits[r, c]);

            double sum_exp = 0.0;
            for (int r = 0; r < R; ++r) {
                double e = std::exp(logits[r, c] - col_max);
                out[r, c] = e;
                sum_exp += e;
            }

            double inv_sum = 1.0 / sum_exp;
            for (int r = 0; r < R; ++r)
                out[r, c] *= inv_sum;
        }
        return out;
    }
}

#endif
