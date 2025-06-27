#include "Matrix.h"
#include <iostream>

int main() {
    Matrix m {5, 7};

    m.at(2,3) = 4;
    m[2, 4] = 5;
    std::cout << m << '\n';

    return 0;
}
