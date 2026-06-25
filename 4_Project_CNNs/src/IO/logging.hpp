//
// Created by Ekkehard Steinmacher on 20.03.25.
//

#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>

// Delete log file if it exists
void deleteLogFile(const std::string& log_file) {
    if (std::remove(log_file.c_str()) != 0) {
        std::cerr << "Failed to delete log file: " << log_file << std::endl;
        return;
    }
    std::cout << "Log file deleted: " << log_file << std::endl;
}

// Create log file if it does not exist
void createLogFile(const std::string& log_file) {
    std::ofstream logStream(log_file, std::ios_base::app); // Open in append mode
    if (!logStream.is_open()) {
        std::cerr << "Failed to create log file: " << log_file << std::endl;
        return;
    }
    logStream.close();
    std::cout << "Log file created: " << log_file << std::endl;
}

void writeToLogFile(const std::string& log_file, const std::string& message) {
    std::ofstream logStream(log_file, std::ios_base::app);
    if (!logStream.is_open()) {
        std::cerr << "Failed to open log file: " << log_file << std::endl;
        return;
    }
    logStream << message << std::endl;
    logStream.close();
}

void writePredictionsToFile(const std::string& log_file, const Eigen::MatrixXd& prediction_tensor,
                            const Eigen::MatrixXd& label_tensor, const int batch_size, const int batch_num) {
    std::ofstream logStream(log_file, std::ios_base::app);
    if (!logStream.is_open()) {
        std::cerr << "Failed to open log file: " << log_file << std::endl;
        return;
    }
    // For each column (image) in the prediction tensor and label tensor write the prediction and the label
    for (int i = 0; i < prediction_tensor.cols(); ++i) {
        int pred, label;
        prediction_tensor.col(i).maxCoeff(&pred);
        label_tensor.col(i).maxCoeff(&label);
        logStream << " - image " << batch_num * batch_size + i << ": Prediction=" << pred << ". Label=" << label << std::endl;
    }
    logStream.close();
}