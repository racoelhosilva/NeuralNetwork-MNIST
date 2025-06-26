#include <iostream>
#include "Matrix.h"

int main() {
    Matrix m {42, 73};

    int r { m.rows() };
    int c { m.cols() };
    std::cout << r << ' ' << c << '\n';

    std::pair<int, int> p { m.shape() };
    std::cout << p.first << ' ' << p.second << '\n';

    return 0;
}
