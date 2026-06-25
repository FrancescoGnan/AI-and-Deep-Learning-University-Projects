//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -I../../eigen -o Loss.exe Loss.cpp

#include <iostream>
#include <Eigen/Dense>

#include "../../src/Optimization/Loss.hpp"

int main() {
    CrossEntropyLoss cel;

    cel.print_info();

    Eigen::MatrixXd pred_tensor(3, 3);
    pred_tensor << .8, .33, 0,
                   .2, .34, 0,
                   .0, .33, 1;
    Eigen::MatrixXd target_tensor(3, 3);
    target_tensor << 1, 0, 0,
                     0, 1, 0,
                     0, 0, 1;
    Eigen::VectorXd loss = *(cel.forward(pred_tensor, target_tensor));

    std::cout << "Loss:\n" << loss << std::endl;

    std::cout << "Backward pass ..." << std::endl;

    Eigen::MatrixXd label_tensor(3, 3);
    label_tensor << 1, 0, 0,
                    0, 1, 0,
                    0, 0, 1;
    Eigen::MatrixXd error = *(cel.backward(label_tensor));

    std::cout << "Error tensor:\n" << error << std::endl;

    return 0;

}