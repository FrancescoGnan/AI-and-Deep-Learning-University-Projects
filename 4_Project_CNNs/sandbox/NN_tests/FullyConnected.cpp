//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Testing the FullyConnected class

// Compile with: g++ -std=c++20 -I../../eigen -o FullyConnected.exe FullyConnected.cpp

#include <iostream>
#include <Eigen/Dense>
#include "../../src/Layers/FullyConnected.hpp"

void change_weights(FullyConnected& fc, Eigen::MatrixXd new_weights) {
    fc.set_weights(new_weights);
}

int main() {
    FullyConnected fc(2, 2, 3, 1);

    fc.print_info();

    fc.print_weights();

    Eigen::MatrixXd new_weights(2, 3);
    new_weights << 0, 2, 1,
                   1, 3, 1;
    change_weights(fc, new_weights);

    fc.print_weights();

    Eigen::MatrixXd input_tensor(2, 3);
    input_tensor << 2, 2, 3,
                    0, 1, 2;
    Eigen::MatrixXd output_tensor = *(fc.forward(input_tensor)); // * for dereferencing the unique_ptr

    std::cout << "Output tensor:\n" << output_tensor << std::endl;

    fc.print_input_tensor();

    std::cout << "Backward pass ..." << std::endl;
    fc.print_weights();
    Eigen::MatrixXd error(2, 3);
    error << 1, 2, 3,
             1, 2, 1;
    Eigen::MatrixXd newError = *(fc.backward(error));

    std::cout << "New error:\n" << newError << std::endl;

    fc.print_weights();

    return 0;
}