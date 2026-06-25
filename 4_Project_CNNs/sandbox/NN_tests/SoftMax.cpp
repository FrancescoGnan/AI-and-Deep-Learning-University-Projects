//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -I../../eigen -o SoftMax.exe SoftMax.cpp

#include <iostream>
#include <Eigen/Dense>

#include "../../src/Layers/SoftMax.hpp"

int main() {
    SoftMax sm;

    sm.print_info();

    Eigen::MatrixXd input_tensor(3, 3);
    input_tensor << 1, 1, 1,
                    1, 1, 1,
                    1, 2, 1;
    Eigen::MatrixXd output_tensor = *(sm.forward(input_tensor)); // * for dereferencing the unique_ptr

    std::cout << "Output tensor:\n" << output_tensor << std::endl;

    std::cout << "Backward pass ..." << std::endl;

    Eigen::MatrixXd error(3, 3);
    error << 1, 2, 3,
             1, 2, 1,
             1, 1, 1;
    Eigen::MatrixXd newError = *(sm.backward(error));

    std::cout << "New error:\n" << newError << std::endl;

    return 0;
}