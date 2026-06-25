// Cross-Entropy Loss

#pragma once

#include <Eigen/Dense>

class CrossEntropyLoss
{
    private:
        Eigen::MatrixXd prediction_tensor; // Important for backward pass
        double eps; // small value to avoid log(0); close to machine epsilon

    public:
        // Constructor
        CrossEntropyLoss() : eps(1e-20) {
          std::cout << "The eps is set to: " << eps << std::endl;
        }

        // Destructor
        ~CrossEntropyLoss() {
            std::cout << "CrossEntropyLoss Layer destroyed" << std::endl;
        }

        // Functions
        std::unique_ptr<Eigen::VectorXd> forward(const Eigen::MatrixXd& pred_tensor, const Eigen::MatrixXd& target_tensor) {
            // -sum(y * log(y_hat))
            // Note: y is the target tensor (contains ground truth) and y_hat is the prediction tensor

            prediction_tensor = pred_tensor.array() + eps;

            Eigen::VectorXd loss = - (target_tensor.array() * prediction_tensor.array().log()).colwise().sum();
            return std::make_unique<Eigen::VectorXd>(loss);
        }

        std::unique_ptr<Eigen::MatrixXd> backward(const Eigen::MatrixXd& label_tensor) {
            // e_n[i] = -y[i]/y_hat[i]
            // Note: e_n is the error tensor and y is the label tensor (contains ground truth) and y_hat is the prediction tensor

            // OPTION 1 -> like on exercise sheet: Changes are in backward of SoftMax and Loss:
            // En = -y / (y_hat + eps)
            Eigen::MatrixXd error_tensor = - label_tensor.array() / prediction_tensor.array();

            // OPTION 2:
//            Eigen::MatrixXd error_tensor = prediction_tensor.array() - label_tensor.array();

            return std::make_unique<Eigen::MatrixXd>(error_tensor);
        }

        void print_info() {
            std::cout << "INFO: CrossEntropyLoss Layer" << std::endl;
        }

};