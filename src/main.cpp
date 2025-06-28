#include "Matrix.h"
#include <iostream>

int main() {
    Matrix m1 {2, 1, 1};
    Matrix m2 {1, 2, 2};

    std::cout << m1 * m2 << '\n';
    std::cout << m2 * m1 << '\n';

    return 0;
}
