// FullyConnected Layer

/* Notes:
* It is a trainable layer, meaning it has its own weights.
* The weights are updated through SGD.
* */

#pragma once

#include <Eigen/Dense>
#include <random>

#include "Base.hpp"
#include "../Optimization/Optimizers.hpp"
#include "../utils/my_math.hpp"

class FullyConnected : public BaseLayer
{
  private:
    Eigen::MatrixXd gradients;
    Eigen::MatrixXd input_tensor; // input tensor + ones for bias
    Eigen::MatrixXd weights;
    int input_size;
    int output_size;
    int batch_size;
    double learning_rate;
    SGD optimizer;

    // Helper function to set weights to random values between 0 and 1
    void initializeWeights() {
      // * dereferences the pointer
        weights = Eigen::MatrixXd::Random(output_size, input_size + 1).array() * 0.5 + 0.5;
    }

	// Xavier Initialization
	void xavier_initialization(Eigen::MatrixXd& weights, int input_size, int output_size) {
    	std::random_device rd;
    	std::mt19937 gen(42);
    	std::normal_distribution<> d(0, std::sqrt(2.0 / (input_size + output_size)));

    	for (int i = 0; i < weights.rows(); ++i) {
        	for (int j = 0; j < weights.cols(); ++j) {
            	weights(i, j) = d(gen);
        	}
    	}
	}

  public:

    // Constructor
    FullyConnected(const int inSize, const int outSize, const int batchSize, const double lr)
        : input_size(inSize),
          output_size(outSize),
          batch_size(batchSize),
          learning_rate(lr),
          optimizer(learning_rate)
    {
        weights.resize(output_size, input_size + 1); // +1 for bias
        gradients.resize(output_size, input_size + 1);
        trainable = true;
        xavier_initialization(weights, input_size, output_size);
    }

    // Destructor
    ~FullyConnected() {
        std::cout << "FullyConnected Layer destroyed" << std::endl;
    }

    // Forward pass
    std::unique_ptr<Eigen::MatrixXd> forward(const Eigen::MatrixXd& in_tensor) override {
        // Operation: MMult: (O x (I+1)) * ((I+1) x B) = (O x B)
        //                 =  weights    * (in_tensor+bias) = output_tensor
        // where O = output_size, I = input_size, B = batch_size

        input_tensor = addRowOfOnes(in_tensor); // Safe input_tensor with ones for backward pass

        auto output_tensor = std::make_unique<Eigen::MatrixXd>(weights * input_tensor);
        return output_tensor;
    }

    // Backward pass
    std::unique_ptr<Eigen::MatrixXd> backward(const Eigen::MatrixXd& error) override {
        // Operation: MMult: ((I+1) x O) * (O x B) = (I+1 x B)
        //                 =  weights^T * error = new_error

        // Calculate gradients
        gradients = error * input_tensor.transpose();
        if (trainable) {
            // Update weights
            weights = optimizer.updateWeights(weights, gradients);
        }
        // Calculate new error
        Eigen::MatrixXd new_error = (weights.transpose() * error);
        // Remove bias row
        return std::make_unique<Eigen::MatrixXd>(removeLastRow(new_error));
    }


    void print_weights() {
        std::cout << "Weights: " << std::endl << weights << std::endl;
    }

    void print_input_tensor() {
        std::cout << "Input tensor: " << std::endl << input_tensor << std::endl;
    }

    void set_weights(Eigen::MatrixXd& new_weights) {
        weights = new_weights;
    }

    void print_info() override {
      std::string str = "INFO: FullyConnected Layer\n";
      str += "\t Input size: " + std::to_string(input_size) + "\n";
      str += "\t Output size: " + std::to_string(output_size) + "\n";
      str += "\t Batch size: " + std::to_string(batch_size) + "\n";
      str += "\t Learning rate: " + std::to_string(learning_rate) + "\n";
      std::cout << str << std::endl;
    }

};