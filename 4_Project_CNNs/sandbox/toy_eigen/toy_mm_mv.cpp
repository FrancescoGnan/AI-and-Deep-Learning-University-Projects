//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -o toy_mm_mv.exe toy_mm_mv.cpp -I../../eigen

#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define arbitrary sized matrix and vector
    // first arg: #rows, second arg: #cols
    Eigen::MatrixXd mat(3, 3);
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;

    Eigen::VectorXd vec(3);
    vec << 1, 2, 3;

    // Perform matrix-vector multiplication
    Eigen::VectorXd result_vec = mat * vec;

    // Output result
    std::cout << "Matrix-Vector Multiplication:\n" << result_vec << "\n\n";

    // Define another arbitrary sized matrix
    Eigen::MatrixXd mat2(3, 3);
    mat2 << 9, 8, 7,
            6, 5, 4,
            3, 2, 1;

    // Perform matrix-matrix multiplication
    Eigen::MatrixXd result_mat = mat * mat2;

    // Output result
    std::cout << "Matrix-Matrix Multiplication:\n" << result_mat << "\n";

    return 0;
}