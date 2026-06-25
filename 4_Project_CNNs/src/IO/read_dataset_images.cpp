#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <stdexcept>

#include "io.hpp"

int main(int argc, char* argv[]) {

  assert(argc == 4);

  try {
    std::cout << "\nC++ parsing file" << std::endl;

    std::cout << "Command executed: " << argv[0];
    for (int i = 0; i < argc; ++i){
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;

    std::string datapath = argv[1];
    std::string filename = argv[2];
    int img_idx = std::stoi(argv[3]);

    std::string filepath = datapath;
    auto images = readIDX3FileImage(filepath);

    std::cout << "Read " << images.size() << " images.\n";
    std::cout << "First image size: " << images[0].rows << "x" << images[0].cols << "\n";

    // Example: Print the first pixel of the first image
    if (!images.empty() && !images[0].pixels.empty()) {
      std::cout << "First pixel value: " << static_cast<int>(images[0].pixels[0]) << "\n";
    }

    saveImageToFile(images[img_idx], filename);

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
  }

  return 0;
}