#include "Matrix.h"
#include <iostream>

double complex(double x) {
    return x > 0 ? 2*x : 0;
}

int main() {
    Matrix m1 {2, 5, 3.7};
    std::cout << m1.apply(complex) << '\n';

    return 0;
}
