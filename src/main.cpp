#include "Matrix.h"
#include <iostream>

int main() {
    Matrix m1 {3, 2, 1};
    Matrix m2 {3, 2, 2};

    std::cout << m1 << '\n';
    std::cout << m2 << '\n';

    std::cout << m1 + m2 << '\n';
    std::cout << m1 - m2 << '\n';

    return 0;
}
