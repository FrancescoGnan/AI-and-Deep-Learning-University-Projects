// Softmax Layer

/* Notes:
* This layer is structurally very similar to the ReLU layer. */

#pragma once

#include <Eigen/Dense>

#include "Base.hpp"

class SoftMax : public BaseLayer
{
    private:
        Eigen::MatrixXd prediction_tensor; // Important for backward pass

    public:
        // Constructor
        SoftMax() = default;

        // Destructor
        ~SoftMax() {
            std::cout << "SoftMax Layer destroyed" << std::endl;
        }

        // Functions
        std::unique_ptr<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input) override {
            // Softmax(x_i) = exp(x_i) / sum(exp(x))
            // Note: Each column of the input tensor belongs to a single image.
            //       The number of columns is the batch size.

            // shift input matrix to avoid numerical instability
            Eigen::VectorXd max_colwise = input.colwise().maxCoeff();
            Eigen::MatrixXd shifted_mat = input.rowwise() - max_colwise.transpose();
            // element-wise exponentiation
            Eigen::MatrixXd exp_mat = shifted_mat.array().exp();
            // sum of each column, i.e. sum of exp(x) for each image
            Eigen::VectorXd sum_colwise = exp_mat.colwise().sum();
            // divide each element by the sum of the corresponding column
            prediction_tensor = exp_mat.array().rowwise() / sum_colwise.array().transpose();

            return std::make_unique<Eigen::MatrixXd>(prediction_tensor);
        }

        std::unique_ptr<Eigen::MatrixXd> backward(const Eigen::MatrixXd& error) override {
            // e_{n-1} = y * (e_n - sum(e_n * y))
            // e_n is the error tensor and y is the prediction tensor

            // OPTION 1: -> like on exercise sheet: Changes are in backward of SoftMax and Loss
            // sum(e_n * y)
            Eigen::VectorXd vec_colsum = (error.array() * prediction_tensor.array()).colwise().sum();
            // e_{n-1}
            Eigen::MatrixXd return_tensor = prediction_tensor.array() * (error.rowwise() - vec_colsum.transpose()).array();

            // Try sigmoiod'(x) = sigmoid(x) * (1 - sigmoid(x))
            // Eigen::MatrixXd return_tensor = prediction_tensor.array() * (1.0 - prediction_tensor.array());

            // OPTION 2
//            Eigen::MatrixXd return_tensor = error;

            return std::make_unique<Eigen::MatrixXd>(return_tensor);
        }

        void print_info() override {
            std::cout << "SoftMax Layer" << std::endl;
        }
};