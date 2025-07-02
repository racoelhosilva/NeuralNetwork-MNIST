#include "Activation.h"
#include <algorithm>
#include <cmath>

namespace activation {
    double ReLU(double val) {
        return val >= 0.0 ? val : 0.0;
    }
    
    double ReLU_prime(double val) {
        return val >= 0.0 ? 1.0 : 0.0;
    }
    
    Matrix softmax(const Matrix& logits) {
        const int rows = logits.rows();
        const int cols = logits.cols();
        Matrix out { rows, cols };
        
        for (int col { 0 }; col < cols; ++col) {
            double col_max = logits[0, col];
            
            for (int row { 1 }; row < rows; ++row) {
                col_max = std::max(col_max, logits[row, col]);
            }
            
            double sum_exp = 0.0;
            for (int row { 0 }; row < rows; ++row) {
                double exp = std::exp(logits[row, col] - col_max);
                out[row, col] = exp;
                sum_exp += exp;
            }
            
            double inverse_sum = 1 / sum_exp;
            for (int row { 0 }; row < rows; ++row) {
                out[row, col] *= inverse_sum;
            }
        }
        return out;
    }
}