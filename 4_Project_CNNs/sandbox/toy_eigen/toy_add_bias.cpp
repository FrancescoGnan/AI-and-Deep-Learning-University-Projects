//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -o toy_add_bias.exe toy_add_bias.cpp -I../../eigen

#include <iostream>
#include <Eigen/Dense>

#include "../../src/utils/my_math.hpp"

int main() {
    // Define an arbitrary-sized matrix
    Eigen::MatrixXd mat(3, 4);
    mat << 1, 2, 3, 4,
           5, 6, 7, 8,
           9, 10, 11, 12;

    std::cout << "Original Matrix:\n" << mat << "\n\n";

    // Add a row of ones
    Eigen::MatrixXd extendedMat = addRowOfOnes(mat);

    std::cout << "Extended Matrix:\n" << extendedMat << "\n";

    Eigen::MatrixXd reducedMat = removeLastRow(extendedMat);

    std::cout << "Reduced Matrix:\n" << reducedMat << "\n";

    return 0;
}