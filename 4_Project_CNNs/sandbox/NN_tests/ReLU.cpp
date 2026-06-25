//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -I../../eigen -o ReLU.exe ReLU.cpp

#include <iostream>
#include <Eigen/Dense>

#include "../../src/Layers/ReLU.hpp"

int main() {
    Eigen::MatrixXd input_tensor(2, 3);
    input_tensor << 1, -2, 3,
                    0, 1, -2;

    std::cout << "Input tensor:\n" << input_tensor << std::endl;

    ReLU relu;
    std::unique_ptr<Eigen::MatrixXd> output_tensor = relu.forward(input_tensor);

    std::cout << "Output tensor:\n" << *output_tensor << std::endl;

    Eigen::MatrixXd error(2, 3);
    error << 1, 2, 3,
             1, 2, 1;
    std::cout << "Error tensor:\n" << error << std::endl;

    std::unique_ptr<Eigen::MatrixXd> newError = relu.backward(error);

    std::cout << "New error:\n" << *newError << std::endl;

    return 0;
}