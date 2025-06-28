#include "Matrix.h"
#include <iostream>

int main() {
    Matrix m1 {3, 2, 1};
    Matrix m2 { m1.transpose() };

    std::cout << m1 << '\n';
    std::cout << m2 << '\n';

    return 0;
}
