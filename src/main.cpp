#include "Matrix.h"
#include <iostream>

int main() {
    Matrix m {42, 73};

    int r { m.rows() };
    int c { m.cols() };
    std::cout << r << ' ' << c << '\n';

    std::pair<int, int> p { m.shape() };
    std::cout << p.first << ' ' << p.second << '\n';

    m.at(2,3) = 4;
    m[2, 4] = 5;
    std::cout << m[2,3] << ' ' << m[2,4] << '\n';

    return 0;
}
