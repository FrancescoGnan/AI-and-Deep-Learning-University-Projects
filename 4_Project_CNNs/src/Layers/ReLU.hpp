// ReLU Layer

/* Notes:
* This layer is not trainable and also has no weights. */

#pragma once

#include <Eigen/Dense>

#include "Base.hpp"

class ReLU : public BaseLayer
{
    private:
        Eigen::MatrixXd prediction_tensor; // Important for backward pass

    public:
        // Constructor; the BaseLayer constructor is actually called automatically, so it would not even be needed here
        ReLU() : BaseLayer() {}

        // Destructor
        ~ReLU() {
            std::cout << "ReLU Layer destroyed" << std::endl;
        }

        // Functions
        std::unique_ptr<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input) override {
            // ReLU(x) = max(0, x)
            prediction_tensor = (input.array() > 0).select(input, 0);
            return std::make_unique<Eigen::MatrixXd>(prediction_tensor);
        }

        std::unique_ptr<Eigen::MatrixXd> backward(const Eigen::MatrixXd& error) override {
            // ReLU'(x) = 1 if x > 0, 0 otherwise
            Eigen::MatrixXd return_tensor = (prediction_tensor.array() > 0).select(error, 0);
            return std::make_unique<Eigen::MatrixXd>(return_tensor);
        }

        void print_prediction_tensor(){
            std::cout << prediction_tensor << std::endl;
        }

        void print_info() override {
            std::cout << "ReLU Layer" << std::endl;
        }
};