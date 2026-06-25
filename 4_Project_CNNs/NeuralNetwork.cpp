//
// Created by Ekkehard Steinmacher on 20.03.25.
//

// Compile with: g++ -std=c++20 -Ieigen -o NeuralNetwork.exe NeuralNetwork.cpp src/IO/io.cpp

#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "src/IO/parsing.hpp"
#include "src/NeuralNetwork.hpp"

int main(int argc, char* argv[]) {

    if(argc != 2){;
        std::cout << "Usage: ./NeuralNetwork.exe <config_file>" << std::endl;
        return 1;
    }

    std::string config_name = argv[1];
    auto config = parseConfigFile(config_name);

    int input_size = 784, output_size = 10;
    int hidden_size, num_epochs, batch_size;
    double learning_rate;
    std::string path_images_train, path_labels_train, path_images_test, path_labels_test, log_file;
    loadConfig(config, num_epochs, hidden_size, batch_size, learning_rate, path_images_train, path_labels_train,
               path_images_test, path_labels_test, log_file);


    NeuralNetwork nn(input_size, hidden_size, output_size, batch_size, learning_rate);

    nn.print_info();

    nn.load_dataset(path_images_train, path_labels_train, path_images_test, path_labels_test);

    deleteLogFile(log_file);
    createLogFile(log_file);
    nn.set_log_file(log_file);

    nn.train(num_epochs);

    nn.validate();

    return 0;
}
