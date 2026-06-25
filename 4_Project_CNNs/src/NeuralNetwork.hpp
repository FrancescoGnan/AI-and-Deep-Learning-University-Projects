// Neural Network Class
// Here the model is implemented

/* Notes:
* The class consists of:
* - a constructor and a destructor
* - a load dataset function
* - a feedforward function
* - a backpropagation function
* - a train function
* - a test/validation function
* - a predict function
* - a print_info function
* */

#pragma once

#include <Eigen/Dense>
#include <utility>

#include "IO/io.hpp"
#include "IO/logging.hpp"

#include "Layers/FullyConnected.hpp"
#include "Layers/ReLU.hpp"
#include "Layers/SoftMax.hpp"

#include "Optimization/Loss.hpp"
#include "Optimization/Optimizers.hpp"

class NeuralNetwork
{
    private:
        // create private members for all needed objects
        int input_size, hidden_size, output_size;
        FullyConnected fc1;
        ReLU relu;
        FullyConnected fc2;
        SoftMax softmax;
        CrossEntropyLoss loss;

        int batch_size;
        double learning_rate;

        std::vector<Image> images_train, images_test, single_image;
        std::vector<Label> labels_train, labels_test, single_label;

        std::string log_file;

    public:
        // Constructor
        NeuralNetwork(int in_size, int hid_size, int out_size, int b_size, double lr)
            : input_size(in_size)
            , hidden_size(hid_size)
            , output_size(out_size)
            , batch_size(b_size)
            , learning_rate(lr)
            // create the layers
            , fc1(in_size, hid_size, b_size, lr)
            , relu()
            , fc2(hid_size, out_size, b_size, lr)
            , softmax()
            , loss()
        {
            std::cout << "Neural Network created" << std::endl;
        }

        // Destructor
        ~NeuralNetwork() {
            std::cout << "Neural Network destroyed" << std::endl;
        }

        // Load dataset
        void load_dataset(const std::string& path_images_train, const std::string& path_labels_train,
                          const std::string& path_images_test, const std::string& path_labels_test)
        {
            images_train = readIDX3FileImage(path_images_train);
            labels_train = readIDX3FileLabel(path_labels_train);
            images_test = readIDX3FileImage(path_images_test);
            labels_test = readIDX3FileLabel(path_labels_test);

            // for debugging
            std::string path_single_image = "../../mnist-datasets/single-image.idx3-ubyte";
            std::string path_single_label = "../../mnist-datasets/single-label.idx1-ubyte";
            single_image = readIDX3FileImage(path_images_train);
            single_label = readIDX3FileLabel(path_labels_train);
        }

        // Set log file
        void set_log_file(std::string file){
            log_file = file;
        }

        // Feedforward
        std::pair<std::unique_ptr<Eigen::VectorXd>, std::unique_ptr<Eigen::MatrixXd>> feedforward(Eigen::MatrixXd& input_tensor, Eigen::MatrixXd& label_tensor)
        {
            // input_tensor: input_size x batch_size; label_tensor: output_size x batch_size

            // forward pass
            auto fc1_out = fc1.forward(input_tensor);
            auto relu_out = relu.forward(*fc1_out);
            auto fc2_out = fc2.forward(*relu_out);
            auto softmax_out = softmax.forward(*fc2_out);
            // compute loss
            auto loss_val = loss.forward(*softmax_out, label_tensor);
            return std::make_pair(std::move(loss_val), std::move(softmax_out));
        }

        // Backpropagation
        void backpropagation(Eigen::MatrixXd& label_tensor)
        {
            // label_tensor: output_size x batch_size

            // compute initial error
            auto initial_error = loss.backward(label_tensor);
            // backward pass
            auto softmax_error = softmax.backward(*initial_error);
            auto fc2_error = fc2.backward(*softmax_error);
            auto relu_error = relu.backward(*fc2_error);
            auto fc1_error = fc1.backward(*relu_error);
        }

        // Train
        void train(int epochs){

            for (int i = 0; i < epochs; i++){

                std::cout << "======================================" << std::endl;
                std::cout << "============= EPOCH: " << i << " ===============" << std::endl;
                std::cout << "======================================" << std::endl;

                Eigen::VectorXd losses;
                Eigen::MatrixXd prediction_tensor;

                int counter = 0, num_iter = 0, remainder = 0;
                int tot_size = images_train.size();
          		remainder = tot_size % batch_size;
                num_iter = (tot_size / batch_size) + ((remainder > 0) ? 1 : 0);
                for (int j = 0; j < num_iter; j++){

                    if (j == num_iter - 1 && remainder > 0){
                        batch_size = remainder;
                    }

                    std::cout << batch_size << std::endl;

                    // Create input_tensor and label_tensor from dataset
                    std::vector<Image> batch_img(images_train.begin() + counter, images_train.begin() + counter + batch_size);
                    Eigen::MatrixXd input_tensor = batchImages(batch_img, batch_size);
                    std::vector<Label> batch_lb(labels_train.begin() + counter, labels_train.begin() + counter + batch_size);
                    Eigen::MatrixXd label_tensor = batchLabels(batch_lb, batch_size);

                    auto [losses, prediction_tensor] = feedforward(input_tensor, label_tensor);
                    double accuracy = compute_accuracy(*prediction_tensor, label_tensor);
                    backpropagation(label_tensor);

                    counter = counter + batch_size;
                    remainder = tot_size - counter;
                    std::cout << tot_size << std::endl;

                    //if (counter % (tot_size / 60) == 0){
                    //    std::cout << "Counter: " << counter << " / " << tot_size << std::endl;
                    //	std::cout << "Loss: " << (*losses).mean() << std::endl;
                    //	std::cout << "Accuracy: " << accuracy << std::endl;
                    //}
                }
            }
        }

        // Accuracy
        double compute_accuracy(Eigen::MatrixXd& prediction_tensor, Eigen::MatrixXd& label_tensor){
            // prediction_tensor: output_size x batch_size; label_tensor: output_size x batch_size
            return (prediction_tensor.array() * label_tensor.array()).sum() / label_tensor.cols();
            // this works because the one-hot encoded label_tensor has only one 1 in each column
        }

        double compute_accuracy_of_most_likely(Eigen::MatrixXd& prediction_tensor, Eigen::MatrixXd& label_tensor){
        	// First turn prediction tensor to one-hot encoded tensor where the 1 is at the prediction with highest likelyhood
            Eigen::MatrixXd one_hot_tensor = one_hot_encode(prediction_tensor);

			return compute_accuracy(one_hot_tensor, label_tensor);
        }

        Eigen::MatrixXd one_hot_encode(const Eigen::MatrixXd& prediction_tensor) {
    		Eigen::MatrixXd one_hot = Eigen::MatrixXd::Zero(prediction_tensor.rows(), prediction_tensor.cols());

    		for (int col = 0; col < prediction_tensor.cols(); ++col) {
        		int max_index;
        		prediction_tensor.col(col).maxCoeff(&max_index);
        		one_hot(max_index, col) = 1.0;
    		}

    		return one_hot;
		}

        // Validation && Logging
		void validate(){

			std::cout << "======================================" << std::endl;
            std::cout << "============ VALIDATION ==============" << std::endl;
            std::cout << "======================================" << std::endl;

            Eigen::VectorXd losses;
            Eigen::MatrixXd prediction_tensor;

			double accuracy_total = 0.0;

            int counter = 0, num_iter = 0, remainder = 0;
            int tot_size = images_test.size();
          	remainder = tot_size % batch_size;
            num_iter = (tot_size / batch_size) + ((remainder > 0) ? 1 : 0);
            for (int j = 0; j < num_iter; j++){

				std::string msg = "Current batch: " + std::to_string(j);
                writeToLogFile(log_file, msg);

                if (j == num_iter - 1 && remainder > 0){
                    batch_size = remainder;
                }

                // Create input_tensor and label_tensor from dataset
                std::vector<Image> batch_img(images_test.begin() + counter, images_test.begin() + counter + batch_size);
                Eigen::MatrixXd input_tensor = batchImages(batch_img, batch_size);
                std::vector<Label> batch_lb(labels_test.begin() + counter, labels_test.begin() + counter + batch_size);
                Eigen::MatrixXd label_tensor = batchLabels(batch_lb, batch_size);

                auto [losses, prediction_tensor] = feedforward(input_tensor, label_tensor);
                double accuracy = compute_accuracy(*prediction_tensor, label_tensor);

                // Write prediction to log file
                writePredictionsToFile(log_file, *prediction_tensor, label_tensor, batch_size, j);

                accuracy_total += accuracy;

                counter = counter + batch_size;
                remainder = tot_size - counter;
			}
            std::cout << "Average accuracy: " << accuracy_total / num_iter << std::endl;
		}

        // Predict
        void predict(int image_number){
			// image_number: 0 - 59999; using training dataset
            image_number = image_number % images_train.size();

        	std::unique_ptr<Eigen::VectorXd> losses;
            std::unique_ptr<Eigen::MatrixXd> prediction_tensor;

        	std::vector<Image> img(images_train.begin() + image_number, images_train.begin() + image_number + 1);
            Eigen::MatrixXd input_tensor = batchImages(img, 1);
            std::vector<Label> lb(labels_train.begin() + image_number, labels_train.begin() + image_number + 1);
            Eigen::MatrixXd label_tensor = batchLabels(lb, 1);

            // Check if label tensor has the correct size
			if (label_tensor.rows() != output_size || label_tensor.cols() != 1) {
    			throw std::runtime_error("Inconsistent label one-hot vector sizes");
			}

            std::tie(losses, prediction_tensor) = feedforward(input_tensor, label_tensor);
            double accuracy = compute_accuracy_of_most_likely(*prediction_tensor, label_tensor);

            std::cout << "Accuracy of image " << image_number << ": " << accuracy << std::endl;
        }


        void print_info() {
            std::string str = "INFO: Neural Network \n";
            str += "\t Input size: " + std::to_string(input_size) + "\n";
            str += "\t Hidden size: " + std::to_string(hidden_size) + "\n";
            str += "\t Output size: " + std::to_string(output_size) + "\n";
            str += "\t Batch size: " + std::to_string(batch_size) + "\n";
            str += "\t Learning rate: " + std::to_string(learning_rate) + "\n";
            std::cout << str << std::endl;
        }
};
