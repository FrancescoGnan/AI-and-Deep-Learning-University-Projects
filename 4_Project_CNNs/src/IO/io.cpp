#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <stdexcept>

#include "io.hpp"

// IMAGE

std::vector<Image> readIDX3FileImage(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + filepath);
  }

  // Read the header
  int32_t magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
  file.read(reinterpret_cast<char*>(&magic_number), 4);
  file.read(reinterpret_cast<char*>(&num_images), 4);
  file.read(reinterpret_cast<char*>(&num_rows), 4);
  file.read(reinterpret_cast<char*>(&num_cols), 4);

  // Convert from big-endian to host byte order
  magic_number = __builtin_bswap32(magic_number);
  num_images = __builtin_bswap32(num_images);
  num_rows = __builtin_bswap32(num_rows);
  num_cols = __builtin_bswap32(num_cols);

  if (magic_number != 2051) {
    throw std::runtime_error("Invalid magic number: " + std::to_string(magic_number));
  }

  std::vector<Image> images;
  images.reserve(num_images);

  // Read each image
  for (int i = 0; i < num_images; ++i) {
    Image img;
    img.rows = num_rows;
    img.cols = num_cols;
    img.pixels.resize(num_rows * num_cols);
    img.values.resize(num_rows * num_cols);

    file.read(reinterpret_cast<char*>(img.pixels.data()), num_rows * num_cols);

    for (int j = 0; j < num_rows * num_cols; ++j){
      img.values[j] = static_cast<double>(img.pixels[j]) / 255.0;
    }

    images.push_back(std::move(img));
  }

  return images;
}

void saveImageToFile(const Image& img, const std::string& filename) {
  /* Takes Image struct as first input */
  const Eigen::VectorXd& values = img.values;
  int rows = img.rows;
  int cols = img.cols;

  // Asserts
  if (values.size() != rows * cols) {
    throw std::invalid_argument("Image size does not match the specified dimensions.");
  }

  // create output stream file
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  // Write image metadata (num_dimensions (=2), num_rows, num_cols)
  file << 2 << std::endl;
  file << rows << std::endl;
  file << cols << std::endl;

  // Write pixels
  for (double v : values){
    file << v << std::endl;
  }

  file.close();
  if (!file) {
    throw std::runtime_error("Failed to write image to file: " + filename);
  }
}

Eigen::MatrixXd batchImages(const std::vector<Image>& images, int batch_size)
{
  if (images.empty())
  {
    throw std::invalid_argument("Image vector is empty");
  }

  int feature_size = images[0].values.size();
  Eigen::MatrixXd batch(feature_size, batch_size);

  for (int i = 0; i < batch_size; ++i)
  {
    if (images[i].values.size() != feature_size)
    {
      throw std::runtime_error("Inconsistent image value sizes");
    }
    batch.col(i) = images[i].values;
  }

  return batch;
}

// LABEL

std::vector<Label> readIDX3FileLabel(const std::string &filepath)
{
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open())
  {
    throw std::runtime_error("Unable to open file: " + filepath);
  }

  // Read the header
  int32_t magic_number = 0, num_labels = 0;
  file.read(reinterpret_cast<char *>(&magic_number), 4);
  file.read(reinterpret_cast<char *>(&num_labels), 4);

  // Convert from big-endian to host byte order
//  if (std::endian::native == std::endian::little)
//  {
//    magic_number = __builtin_bswap32(magic_number);
//    num_labels = __builtin_bswap32(num_labels);
//  }
  // Convert from big-endian to host byte order
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);
  #endif

  if (magic_number != 2049)
  {
    throw std::runtime_error("Invalid magic number: " + std::to_string(magic_number));
  }

  std::vector<Label> labels;
  labels.reserve(num_labels);

  // Read each image
  for (int i = 0; i < num_labels; ++i)
  {
    Label lb;
    char label_char;
    file.read(&label_char, 1);
    lb.label = static_cast<int>(label_char);

    lb.one_hot_vec[lb.label] = 1.0; // change to one-hot vector

    labels.push_back(std::move(lb));
  }

  return labels;
}

void saveLabelToFile(const Label &lb, const std::string &filename)
{
  /* Takes Label struct as first input */
  const Eigen::VectorXd &values = lb.one_hot_vec;

  // Asserts
  if (values.size() != 10)
  {
    throw std::invalid_argument("One-hot vec size does not match the 10 different labels.");
  }

  // create output stream file
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  // Write image metadata (num_labels, num_different_categories)
  file << 1 << std::endl;
  file << 10 << std::endl;

  // Write pixels
  for (double v : values)
  {
    file << v << std::endl;
  }

  file.close();
  if (!file)
  {
    throw std::runtime_error("Failed to write image to file: " + filename);
  }
}

Eigen::MatrixXd batchLabels(const std::vector<Label>& labels, int batch_size)
{
  if (labels.empty())
  {
    throw std::invalid_argument("Label vector is empty");
  }

  int feature_size = labels[0].one_hot_vec.size();
  Eigen::MatrixXd batch(feature_size, batch_size);

  for (int i = 0; i < batch_size; ++i)
  {
    if (labels[i].one_hot_vec.size() != feature_size)
    {
      throw std::runtime_error("Inconsistent label one-hot vector sizes");
    }
    batch.col(i) = labels[i].one_hot_vec;
  }

  return batch;
}