// Parsing the input config file to load the hyperparameters

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <cctype>

// Function to trim spaces from a string
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" ");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" ");
    return str.substr(first, last - first + 1);
}

// Function to parse the configuration file
std::unordered_map<std::string, std::string> parseConfigFile(const std::string& filename) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        // Remove comments
        size_t commentPos = line.find("//");
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        // Trim spaces
        line = trim(line);
        if (line.empty()) continue; // Skip empty lines

        // Parse key-value pair
        size_t delimiterPos = line.find("=");
        if (delimiterPos != std::string::npos) {
            std::string key = trim(line.substr(0, delimiterPos));
            std::string value = trim(line.substr(delimiterPos + 1));
            config[key] = value;
        }
    }
    return config;
}

// Function to load config to variables
void loadConfig(const std::unordered_map<std::string, std::string>& config,
                int& num_epochs, int& hidden_size, int& batch_size, double& learning_rate,
                std::string& path_images_train, std::string& path_labels_train,
                std::string& path_images_test, std::string& path_labels_test, std::string& log_file) {

    num_epochs = std::stoi(config.at("num_epochs"));
    hidden_size = std::stoi(config.at("hidden_size"));
    batch_size = std::stoi(config.at("batch_size"));
    learning_rate = std::stod(config.at("learning_rate"));

    path_images_train = config.at("rel_path_train_images");
    path_labels_train = config.at("rel_path_train_labels");
    path_images_test = config.at("rel_path_test_images");
    path_labels_test = config.at("rel_path_test_labels");
    log_file = config.at("rel_path_log_file");
}
