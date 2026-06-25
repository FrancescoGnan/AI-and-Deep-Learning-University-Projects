#ifndef IO_HPP
#define IO_HPP

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdint>

// IMAGE

struct Image
{
    std::vector<uint8_t> pixels; // Stores pixels
    Eigen::VectorXd values;  // 0 - 255 converted to 0.0 - 1.0
    int rows;
    int cols;
};

std::vector<Image> readIDX3FileImage(const std::string &filepath);
void saveImageToFile(const Image &img, const std::string &filename);

Eigen::MatrixXd batchImages(const std::vector<Image>& images, int batch_size);

// LABEL

struct Label
{
    int label;                      // value 0-9
    Eigen::VectorXd one_hot_vec;     // store labels as one-hot encoded

    Label() : one_hot_vec(10) {      // Constructor -> 10 elements all initialized to 0.0
        one_hot_vec.setZero();
    }
};

std::vector<Label> readIDX3FileLabel(const std::string &filepath);
void saveLabelToFile(const Label &lb, const std::string &filename);

Eigen::MatrixXd batchLabels(const std::vector<Label>& labels, int batch_size);

#endif // IO_HPP