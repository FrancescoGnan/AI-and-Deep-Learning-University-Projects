#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Matrix2d A, B, C;

    A << 1, 2,
         3, 4;
    B << 5, 6,
         7, 8;

    C = A + B;

    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Matrix B:\n" << B << "\n\n";
    std::cout << "A + B =\n" << C << "\n";

    return 0;
}