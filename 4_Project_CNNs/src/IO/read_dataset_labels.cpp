#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <bit>

#include "io.hpp"

int main(int argc, char *argv[])
{

  assert(argc == 4);

  try
  {
    std::cout << "\nC++ parsing file" << std::endl;

    std::cout << "Command executed: " << argv[0];
    for (int i = 1; i < argc; ++i)
    {
      std::cout << " " << argv[i];
    }
    std::cout << std::endl;

    std::string datapath = argv[1];
    std::string filename = argv[2];
    int lb_idx = std::stoi(argv[3]);

    std::string filepath = datapath;
    auto labels = readIDX3FileLabel(filepath);

    std::cout << "Read " << labels.size() << " labels.\n";

    saveLabelToFile(labels[lb_idx], filename);
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error: " << e.what() << "\n";
  }

  return 0;
}