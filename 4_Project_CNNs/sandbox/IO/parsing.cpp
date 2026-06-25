//
// Created by Ekkehard Steinmacher on 19.03.25.
//

// Compile with: g++ -std=c++20 -o parsing.exe parsing.cpp

#include <iostream>

#include "../../src/IO/parsing.hpp"

int main(int argc, char* argv[]) {

    if(argc != 2){;
        std::cout << "Usage: ./parsing.exe <config_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    auto config = parseConfigFile(filename);

    // Print parsed configuration
    for (const auto& entry : config) {
        std::cout << entry.first << " = " << entry.second << std::endl;
    }

    return 0;
}