// Stochasic Gradient Descent (SGD) is our Optimizer to update the weights of the FullyConnected Layer.

#pragma once

#include <Eigen/Dense>

class SGD {
    private:
        double learning_rate;

    public:
        SGD(double lr) : learning_rate(lr) { std::cout << "SGD Optimizer created with learning rate: " << learning_rate << std::endl; }
        ~SGD() { std::cout << "SGD Optimizer destructed" << std::endl; } // = default;

        void setLearningRate(double lr) {
            learning_rate = lr;
        }

        double getLearningRate() {
            return learning_rate;
        }

        Eigen::MatrixXd updateWeights(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients) {
            weights = weights - learning_rate * gradients;
            return weights;
        }
};