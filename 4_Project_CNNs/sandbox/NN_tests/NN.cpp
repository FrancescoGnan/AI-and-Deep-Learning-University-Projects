//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -I../../eigen -o NN.exe NN.cpp ../../src/IO/io.cpp

#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "../../src/IO/parsing.hpp"
#include "../../src/NeuralNetwork.hpp"

int main(int argc, char* argv[]) {

    if(argc != 2){;
        std::cout << "Usage: ./NN.exe <config_file>" << std::endl;
        return 1;
    }

    std::string config_name = argv[1];
    auto config = parseConfigFile(config_name);

    // Print parsed configuration
    for (const auto& entry : config) {
        std::cout << entry.first << " = " << entry.second << std::endl;
    }

    int input_size = 784, output_size = 10;
    int hidden_size, num_epochs, batch_size;
    double learning_rate;
    std::string path_images_train, path_labels_train, path_images_test, path_labels_test, log_file;
    loadConfig(config, num_epochs, hidden_size, batch_size, learning_rate, path_images_train, path_labels_train,
               path_images_test, path_labels_test, log_file);


    NeuralNetwork nn(input_size, hidden_size, output_size, batch_size, learning_rate);

    nn.print_info();

//    std::string path_images_train = "../../mnist-datasets/train-images.idx3-ubyte";
//    std::string path_labels_train = "../../mnist-datasets/train-labels.idx1-ubyte";
//    std::string path_images_test = "../../mnist-datasets/t10k-images.idx3-ubyte";
//    std::string path_labels_test = "../../mnist-datasets/t10k-labels.idx1-ubyte";
//
//    std::string log_file = "../../log_test.txt";

    nn.load_dataset(path_images_train, path_labels_train, path_images_test, path_labels_test);

    deleteLogFile(log_file);
    createLogFile(log_file);
    nn.set_log_file(log_file);

//    std::unique_ptr<Eigen::VectorXd> losses;
//    Eigen::MatrixXd input_tensor(3, 3); //(input_size, batch_size);
//    Eigen::MatrixXd label_tensor(output_size, batch_size);
//    input_tensor << 1, 2, 3,
//                    4, 5, 6,
//                    7, 8, 9;
//    label_tensor << 1, 0, 0,
//                    0, 1, 0,
//                    0, 0, 1;
//    losses = nn.feedforward(input_tensor, label_tensor);
//
//    std::cout << "Losses: " << *losses << std::endl;
//
//    // backpropagation
//    for (int i = 0; i < 10; ++i){
//        nn.backpropagation(label_tensor);
//        losses = nn.feedforward(input_tensor, label_tensor);
//        std::cout << "Losses: " << *losses << std::endl;
//    }

//    nn.train(1);

    nn.validate();

    // Create a random number generator
//    std::random_device rd;  // Seed
//    std::mt19937 gen(rd()); // Mersenne Twister engine
//    std::uniform_int_distribution<> dis(0, 60000); // Define the range
//
//    for (int i = 0; i < 10; ++i){
//        nn.predict(dis(gen));
//    }



    return 0;
}
